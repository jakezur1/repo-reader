document.addEventListener('DOMContentLoaded', function () {
    const readmeSection = document.querySelector('div#readme');
    if (!readmeSection) {
      const container = document.querySelector('.repository-content');
      if (container) {
        const button = document.createElement('button');
        button.textContent = 'Generate README';
        button.onclick = generateReadme;
        container.prepend(button);
      }
    }
  });
  
  function generateReadme() {
    // Logic to generate README or call to your server/api to generate it
    console.log('Generating README...');
  }