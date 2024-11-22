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

let uploadedImageFile = null;

function previewImage(event) {
    uploadedImageFile = event.target.files[0];
    let reader = new FileReader();
    reader.onload = function(){
        let output = document.getElementById('image_preview');
        output.src = reader.result;
    };
    reader.readAsDataURL(uploadedImageFile);
}

// function detectFaces() {
//     if (!uploadedImageFile) {
//         alert("Please upload an image first.");
//         return;
//     }

//     let formData = new FormData();
//     formData.append('image', uploadedImageFile);

//     fetch('/detect_embedding', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => response.blob())
//     .then(blob => {
//         let img = document.getElementById('image_result');
//         img.src = URL.createObjectURL(blob);
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
// }
function detectFaces() {
if (!uploadedImageFile) {
      alert("Please upload an image first.");
      return;
  }

  let formData = new FormData();
  formData.append('image', uploadedImageFile);

  fetch('/detect_embedding', {
      method: 'POST',
      body: formData
  })
  .then(response => {
      if (!response.ok) {
          return response.json().then(err => { throw err; });
      }
      return response.json();
  })
  .then(data => {
      // Display the result image
      let imgResult = document.getElementById('image_result');
      imgResult.src = 'data:image/jpeg;base64,' + data.result_image;

      // Clear previous face data
      let facesContainer = document.getElementById('faces_container');
      facesContainer.innerHTML = '';

      // Loop through each face and display information
      data.faces.forEach(face => {
          // Create a container for each face
          let faceDiv = document.createElement('div');
          faceDiv.className = 'face-info';

          // Face image
          let faceImg = document.createElement('img');
          faceImg.src = 'data:image/jpeg;base64,' + face.face_image;
          faceImg.alt = `Face ${face.face_index}`;
          faceImg.style.maxWidth = '150px';

          // Embedding (display first few elements for brevity)
          let embeddingP = document.createElement('p');
          embeddingP.textContent = `Embedding: [${face.embedding.slice(0, 5).map(e => e.toFixed(4)).join(', ')}... ]`;

          // Emotion label
          let emotionP = document.createElement('p');
          emotionP.textContent = `Emotion: ${face.emotion}`;

          // Append to faceDiv
          faceDiv.appendChild(faceImg);
          faceDiv.appendChild(embeddingP);
          faceDiv.appendChild(emotionP);

          // Append faceDiv to facesContainer
          facesContainer.appendChild(faceDiv);
      });
  })
  .catch(error => {
      console.error('Error:', error);
      alert(error.error || 'An error occurred during face detection.');
  });
}
