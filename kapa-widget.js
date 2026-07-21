(function () {
  var websiteId = '3c6baf90-a547-43ae-9431-03e31215bf6b';

  if (document.querySelector('script[data-website-id="' + websiteId + '"]')) return;

  var script = document.createElement('script');
  script.src = 'https://widget.kapa.ai/kapa-widget.bundle.js';
  script.async = true;
  script.setAttribute('data-website-id', websiteId);
  script.setAttribute('data-project-name', 'Weights & Biases');
  script.setAttribute('data-project-color', '#FCBC32');
  script.setAttribute(
    'data-project-logo',
    'https://site.wandb.ai/wp-content/uploads/2024/05/pictorial-mark-black-1.svg'
  );
  script.setAttribute('data-modal-image', 'https://docs.wandb.ai/icons/wandb-gold.svg');
  script.setAttribute('data-color-scheme-selector', '.dark');
  // Keep anonymous Kapa tracking off unless it is wired into the site's consent flow.
  script.setAttribute('data-user-analytics-cookie-enabled', 'false');

  document.head.appendChild(script);
})();
