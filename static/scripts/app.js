// Hàm mở camera
function openCamera() {
    const cameraView = document.getElementById("camera_view");
  
    if (cameraView) {
      // Kiểm tra nếu iframe chưa được thêm, thêm iframe mới vào DOM
      if (!cameraView.querySelector("iframe")) {
        const iframe = document.createElement("iframe");
        iframe.src = "/camera_user"; // Route để phát video stream
        iframe.width = "640";
        iframe.height = "480";
        iframe.frameBorder = "0";
  
        cameraView.appendChild(iframe);
      }
      cameraView.style.display = "block"; // Hiển thị khu vực camera
    } else {
      console.error("Camera view element not found.");
    }
  }
  
  // Hàm đóng camera
  function closeCamera() {
    const cameraView = document.getElementById("camera_view");
  
    if (cameraView) {
      cameraView.style.display = "none"; // Ẩn khu vực camera
      cameraView.innerHTML = ""; // Xóa iframe để dừng stream
    } else {
      console.error("Camera view element not found.");
    }
  }
  