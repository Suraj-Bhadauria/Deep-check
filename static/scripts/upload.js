document.getElementById('uploadBtn').addEventListener('click', function(){
  const fileInput = document.getElementById('fileInput');
  const resultDiv = document.getElementById('result');
  
  if(fileInput.files.length === 0) {
    resultDiv.innerHTML = "<p style='color: red;'>Please select a file to upload.</p>";
    return;
  }
  
  // Simulate file processing
  resultDiv.innerHTML = "<p>Processing file...</p>";
  
  setTimeout(() => {
    // Simulate a detection result (for demonstration purposes)
    const isDeepfake = Math.random() < 0.5;
    if(isDeepfake) {
      resultDiv.innerHTML = "<p style='color: #ff4d4d;'>Warning: Deepfake detected!</p>";
    } else {
      resultDiv.innerHTML = "<p style='color: #4dff88;'>File appears to be genuine.</p>";
    }
  }, 2000);
});
