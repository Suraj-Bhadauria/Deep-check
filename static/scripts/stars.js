document.addEventListener('DOMContentLoaded', function() {
  const canvas = document.getElementById('starfield');
  if (!canvas) {
    console.error('Canvas element not found!');
    return;
  }
  const ctx = canvas.getContext('2d');
  const numStars = 150;  // Number of stars to create
  let stars = [];

  // Set the canvas size to match the window dimensions
  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    console.log('Canvas dimensions:', canvas.width, canvas.height);
  }

  // Initialize the stars with random properties
  function initStars() {
    stars = [];
    for (let i = 0; i < numStars; i++) {
      stars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        radius: Math.random() * 3 + 1,      // Random radius between 1 and 4
        speed: Math.random() * 0.5 + 0.2,     // Random speed between 0.2 and 0.7
        alpha: Math.random() * 0.5 + 0.5      // Random opacity between 0.5 and 1.0
      });
    }
  }

  // Draw each star onto the canvas
  function drawStars() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stars.forEach(star => {
      ctx.beginPath();
      ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${star.alpha})`;
      ctx.fill();
    });
  }

  // Update each star's position for the animation
  function updateStars() {
    stars.forEach(star => {
      star.y += star.speed;
      // If a star moves beyond the bottom, reset it to the top at a random x position
      if (star.y > canvas.height) {
        star.y = 0;
        star.x = Math.random() * canvas.width;
      }
    });
  }

  // Animation loop: draw and update stars, then request the next frame
  function animateStars() {
    drawStars();
    updateStars();
    requestAnimationFrame(animateStars);
  }

  // Initialize canvas, stars, and start the animation
  resizeCanvas();
  initStars();
  animateStars();

  // Update canvas size and reinitialize stars on window resize
  window.addEventListener('resize', () => {
    resizeCanvas();
    initStars();
  });
});
