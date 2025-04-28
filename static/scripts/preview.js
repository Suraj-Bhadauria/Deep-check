function goToResultPage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();
        
        reader.onload = function (e) {
            localStorage.setItem("uploadedImage", e.target.result); // Store Base64 Image
            window.location.href = "./image.html"; // Redirect to result page
        };
        
        reader.readAsDataURL(file); // Convert image to Base64
    } else {
        alert("Please select an image!");
    }
}