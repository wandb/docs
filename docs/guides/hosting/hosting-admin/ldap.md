# LDAP

## Prerequisites
Before you get started, ensure you have satisfied the following prerequisites:

1. Install Docker and create a Web App that interacts with an LDAP server.


## How to bind your LDAP server securely

The following section demonstrates how to set up LDAP server and how to configure your LDAP client in W&B Server.

### Set up your LDAP sever
1. First, run the LDAP server backend with Docker: 

```bash
docker run \
 --hostname ldap-service.example.org \
 --name ldap-service -p 6636:636 -p 6389:389 --detach osixia/openldap:1.5.0
```

2. Start the web UI

```bash
docker run \
 -p 6443:443 --name phpldapadmin-service \
 --hostname phpldapadmin-service --link ldap-service:ldap-host \
 --env PHPLDAPADMIN_LDAP_HOSTS=ldap-host --detach osixia/phpldapadmin:0.9.0
```

You should see two docker containers running within your Docker Desktop: