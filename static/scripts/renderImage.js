window.onload = function () {
    const storedImage = localStorage.getItem("uploadedImage");
    
    if (storedImage) {
        document.getElementById("resultImage").src = storedImage; // Load image
    } else {
        document.querySelector(".image").innerHTML = "<p class='text-center text-white'>No image found. Please go back and upload one.</p>";
    }
};