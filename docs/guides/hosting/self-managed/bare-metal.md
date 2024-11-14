---
title: Deploy W&B Platform On-premises
description: Hosting W&B Server on on-premises infrastructure
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::info
W&B recommends fully managed deployment options such as [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) or [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) deployment types. W&B fully managed services are simple and secure to use, with minimum to no configuration required.
:::


Reach out to the W&B Sales Team for related question: [contact@wandb.com](mailto:contact@wandb.com).

## Infrastructure guidelines

Please read through our [reference architecture](./ref-arch.md) before you start deploying W&B - especially the infrastructure requirements.


## MySQL database

There are a number of enterprise services that make operating a scalable MySQL database simpler. W&B recommends looking into one of the following solutions:

[https://www.percona.com/software/mysql-database/percona-server](https://www.percona.com/software/mysql-database/percona-server)

[https://github.com/mysql/mysql-operator](https://github.com/mysql/mysql-operator)

Satisfy the conditions below if you run W&B Server MySQL 8.0 or when you upgrade from MySQL 5.7 to 8.0:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
```

Due to some changes in the way that MySQL 8.0 handles `sort_buffer_size`, you might need to update the `sort_buffer_size` parameter from its default value of `262144`. Our recommendation is to set the value to `67108864(64MiB)` in order for the database to efficiently work with the W&B application. Note that, this only works with MySQL versions 8.0.28 and above.

### Database considerations


Create a database and a user with the following SQL query. Replace `SOME_PASSWORD` with password of your choice:

```sql
CREATE USER 'wandb_local'@'%' IDENTIFIED BY 'SOME_PASSWORD';
CREATE DATABASE wandb_local CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL ON wandb_local.* TO 'wandb_local'@'%' WITH GRANT OPTION;
```

#### Parameter group configuration

Ensure that the following parameter groups are set to tune the database performance:

```
binlog_format = 'ROW'
innodb_online_alter_log_max_size = 268435456
sync_binlog = 1
innodb_flush_log_at_trx_commit = 1
binlog_row_image = 'MINIMAL'
sort_buffer_size = 67108864
```

## Object store
The object store can be externally hosted on a [Minio cluster](https://docs.min.io/minio/k8s/), or any Amazon S3 compatible object store that has support for signed URLs. Run the [following script](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b) to check if your object store supports signed URLs.

Additionally, the following CORS policy needs to be applied to the object store.

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<CORSConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<CORSRule>
    <AllowedOrigin>http://YOUR-W&B-SERVER-IP</AllowedOrigin>
    <AllowedMethod>GET</AllowedMethod>
    <AllowedMethod>PUT</AllowedMethod>
    <AllowedMethod>HEAD</AllowedMethod>
    <AllowedHeader>*</AllowedHeader>
</CORSRule>
</CORSConfiguration>
```

You can specify your credentials in a connection string when you connect to an Amazon S3 compatible object store. For example, you can specify the following: 

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

You can optionally tell W&B to only connect over TLS if you configure a trusted SSL certificate for your object store. To do so, add the `tls` query parameter to the URL. For example, the following URL example demonstrates how to add the TLS query parameter to an Amazon S3 URI:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```

:::caution
This will only work if the SSL certificate is trusted. W&B does not support self-signed certificates.
:::

Set `BUCKET_QUEUE` to `internal://` if you use third-party object stores. This tells the W&B server to manage all object notifications internally instead of depending on an external SQS queue or equivalent.

The most important things to consider when running your own object store are:

1. **Storage capacity and performance**. It's fine to use magnetic disks, but you should be monitoring the capacity of these disks. Average W&B usage results in 10's to 100's of Gigabytes. Heavy usage could result in Petabytes of storage consumption.
2. **Fault tolerance.** At a minimum, the physical disk storing the objects should be on a RAID array. If you use minio, consider running it in [distributed mode](https://docs.min.io/minio/baremetal/installation/deploy-minio-distributed.html#deploy-minio-distributed).
3. **Availability.** Monitoring should be configured to ensure the storage is available.

There are many enterprise alternatives to running your own object storage service such as:

1. [https://aws.amazon.com/s3/outposts/](https://aws.amazon.com/s3/outposts/)
2. [https://www.netapp.com/data-storage/storagegrid/](https://www.netapp.com/data-storage/storagegrid/)

### MinIO set up

If you use minio, you can run the following commands to create a bucket.

```bash
mc config host add local http://$MINIO_HOST:$MINIO_PORT "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api s3v4
mc mb --region=us-east1 local/local-files
```

## Deploy W&B Server application to Kubernetes

The recommended installation method is with the official W&B Helm chart. Follow [this section](../operator#deploy-wb-with-helm-cli) to deploy the W&B Server application.


### OpenShift

W&B supports operating from within an [OpenShift Kubernetes cluster](https://www.redhat.com/en/technologies/cloud-computing/openshift). 

:::info
W&B recommends you install with the official W&B Helm chart. 
:::
#### Run the container as an un-privileged user

By default, containers use a `$UID` of 999. Specify `$UID` >= 100000 and a `$GID` of 0 if your orchestrator requires the container run with a non-root user.

:::note
W&B  must start as the root group (`$GID=0`) for file system permissions to function properly.
:::

An example security context for Kubernetes looks similar to the following:

```
spec:
  securityContext:
    runAsUser: 100000
    runAsGroup: 0
```

## Networking

### Load balancer

Run a load balancer that stop network requests at the appropriate network boundary. 

Common load balancers include:
1. [Nginx Ingress](https://kubernetes.github.io/ingress-nginx/)
2. [Istio](https://istio.io)
3. [Caddy](https://caddyserver.com)
4. [Cloudflare](https://www.cloudflare.com/load-balancing/)
5. [Apache](https://www.apache.org)
6. [HAProxy](http://www.haproxy.org)

Ensure that all machines used to execute machine learning payloads, and the devices used to access the service through web browsers, can communicate to this endpoint. 


### SSL / TLS

W&B Server does not stop SSL. If your security policies require SSL communication within your trusted networks consider using a tool like Istio and [side car containers](https://istio.io/latest/docs/reference/config/networking/sidecar/). The load balancer itself should terminate SSL with a valid certificate. Using self-signed certificates is not supported and will cause a number of challenges for users. If possible using a service like [Let's Encrypt](https://letsencrypt.org) is a great way to provided trusted certificates to your load balancer. Services like Caddy and Cloudflare manage SSL for you.

### Example nginx configuration

The following is an example configuration using nginx as a reverse proxy.

```nginx
events {}
http {
    # If we receive X-Forwarded-Proto, pass it through; otherwise, pass along the
    # scheme used to connect to this server
    map $http_x_forwarded_proto $proxy_x_forwarded_proto {
        default $http_x_forwarded_proto;
        ''      $scheme;
    }

    # Also, in the above case, force HTTPS
    map $http_x_forwarded_proto $sts {
        default '';
        "https" "max-age=31536000; includeSubDomains";
    }

    # If we receive X-Forwarded-Host, pass it though; otherwise, pass along $http_host
    map $http_x_forwarded_host $proxy_x_forwarded_host {
        default $http_x_forwarded_host;
        ''      $http_host;
    }

    # If we receive X-Forwarded-Port, pass it through; otherwise, pass along the
    # server port the client connected to
    map $http_x_forwarded_port $proxy_x_forwarded_port {
        default $http_x_forwarded_port;
        ''      $server_port;
    }

    # If we receive Upgrade, set Connection to "upgrade"; otherwise, delete any
    # Connection header that may have been passed to this server
    map $http_upgrade $proxy_connection {
        default upgrade;
        '' close;
    }

    server {
        listen 443 ssl;
        server_name         www.example.com;
        ssl_certificate     www.example.com.crt;
        ssl_certificate_key www.example.com.key;

        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $proxy_connection;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $proxy_x_forwarded_proto;
        proxy_set_header X-Forwarded-Host $proxy_x_forwarded_host;

        location / {
            proxy_pass  http://$YOUR_UPSTREAM_SERVER_IP:8080/;
        }

        keepalive_timeout 10;
    }
}
```

## Verify your installation

Very your W&B Server is configured properly. Run the following commands in your terminal:

```bash
pip install wandb
wandb login --host=https://YOUR_DNS_DOMAIN
wandb verify
```

Check log files to view any errors the W&B Server hits at startup. Run the following commands based on whether if you use Docker or Kubernetes: 


```bash
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

Contact W&B Support if you encounter errors. 