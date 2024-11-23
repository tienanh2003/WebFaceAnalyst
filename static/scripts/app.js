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
  reader.onload = function () {
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
// function detectFaces() {
//   if (!uploadedImageFile) {
//     alert("Please upload an image first.");
//     return;
//   }

//   let formData = new FormData();
//   formData.append('image', uploadedImageFile);

//   fetch('/detect_embedding', {
//     method: 'POST',
//     body: formData
//   })
//     .then(response => {
//       if (!response.ok) {
//         return response.json().then(err => { throw err; });
//       }
//       return response.json();
//     })
//     .then(data => {
//       // Display the result image
//       let imgResult = document.getElementById('image_result');
//       imgResult.src = 'data:image/jpeg;base64,' + data.result_image;

//       // Clear previous face data
//       let facesContainer = document.getElementById('faces_container');
//       facesContainer.innerHTML = '';

//       // Loop through each face and display information
//       data.faces.forEach(face => {
//         // Create a container for each face
//         let faceDiv = document.createElement('div');
//         faceDiv.className = 'face-info';

//         // Face image
//         let faceImg = document.createElement('img');
//         faceImg.src = 'data:image/jpeg;base64,' + face.face_image;
//         faceImg.alt = `Face ${face.face_index}`;
//         faceImg.style.maxWidth = '150px';

//         // Embedding (display first few elements for brevity)
//         let embeddingP = document.createElement('p');
//         embeddingP.textContent = `Embedding: [${face.embedding.slice(0, 5).map(e => e.toFixed(4)).join(', ')}... ]`;

//         // Emotion label
//         let emotionP = document.createElement('p');
//         emotionP.textContent = `Emotion: ${face.emotion}`;

//         // Append to faceDiv
//         faceDiv.appendChild(faceImg);
//         faceDiv.appendChild(embeddingP);
//         faceDiv.appendChild(emotionP);

//         // Append faceDiv to facesContainer
//         facesContainer.appendChild(faceDiv);
//       });
//     })
//     .catch(error => {
//       console.error('Error:', error);
//       alert(error.error || 'An error occurred during face detection.');
//     });
// }

let uploadedVideoFile = null;

// Load video
function uploadVideo() {
  const input = document.getElementById("video_input");
  const videoElement = document.getElementById("uploaded_video");

  if (input.files.length === 0) {
    alert("Please upload a video file.");
    return;
  }

  uploadedVideoFile = input.files[0];
  const url = URL.createObjectURL(uploadedVideoFile);
  videoElement.src = url;
}

function previewVideo(event) {
  uploadedVideoFile = event.target.files[0];
  let videoPreview = document.getElementById('video_preview');
  videoPreview.src = URL.createObjectURL(uploadedVideoFile);
  videoPreview.load();
}

// Load video preview
function uploadVideo() {
  const input = document.getElementById("video_input");
  const videoPreview = document.getElementById("video_preview");

  if (!input.files.length) {
    alert("Please select a video file to upload.");
    return;
  }

  const file = input.files[0];
  const url = URL.createObjectURL(file);

  videoPreview.src = url;
  videoPreview.load();
}

function detectVideo() {
  const fileInput = document.getElementById("video_input");
  const file = fileInput.files[0];
  if (!file) {
    alert("Vui lòng tải lên một video.");
    return;
  }

  const formData = new FormData();
  formData.append("video", file);

  fetch("/detect_video", {
    method: "POST",
    body: formData,
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        return;
      }

      // Gán URL của video xử lý vào thẻ <video>
      const videoResult = document.getElementById("video_result");
      videoResult.src = data.video_url;
      videoResult.load();
    })
    .catch(error => {
      console.error("Lỗi:", error);
      alert("Đã xảy ra lỗi trong quá trình xử lý video.");
    });
}

function addUser() {
  const usernameInput = document.getElementById("usernameInput");
  const profileImageUpload = document.getElementById("profileImageUpload");

  if (!usernameInput.value || !profileImageUpload.files.length) {
    alert("Please enter a name and upload an image.");
    return;
  }

  const formData = new FormData();
  formData.append("name", usernameInput.value);
  formData.append("image", profileImageUpload.files[0]);

  fetch("/add_user", {
    method: "POST",
    body: formData,
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        return;
      }
      alert("User added successfully!");
    })
    .catch(error => {
      console.error("Error:", error);
      alert("An error occurred while adding the user.");
    });
}

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
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        return;
      }

      // Hiển thị ảnh kết quả
      let imgResult = document.getElementById('image_result');
      imgResult.src = 'data:image/jpeg;base64,' + data.result_image;

      // Hiển thị thông tin khuôn mặt
      let facesContainer = document.getElementById('faces_container');
      facesContainer.innerHTML = '';

      data.faces.forEach(face => {
        let faceDiv = document.createElement('div');
        faceDiv.className = 'face-info';

        // Ảnh khuôn mặt
        let faceImg = document.createElement('img');
        faceImg.src = 'data:image/jpeg;base64,' + face.face_image;
        faceImg.alt = `Face ${face.face_index}`;
        faceImg.style.maxWidth = '150px';

        // Thông tin nhận diện
        let nameP = document.createElement('p');
        nameP.textContent = `Name: ${face.name}`;

        // Khoảng cách (nếu có)
        let distanceP = document.createElement('p');
        distanceP.textContent = `Distance: ${face.distance}`;

        // Cảm xúc
        let emotionP = document.createElement('p');
        emotionP.textContent = `Emotion: ${face.emotion}`;

        // Thêm các thông tin vào div khuôn mặt
        faceDiv.appendChild(faceImg);
        faceDiv.appendChild(nameP);
        faceDiv.appendChild(distanceP);
        faceDiv.appendChild(emotionP);

        // Thêm khuôn mặt vào danh sách
        facesContainer.appendChild(faceDiv);
      });
    })
    .catch(error => {
      console.error('Error:', error);
      alert('An error occurred during face detection.');
    });
}

function previewUploadedImage(event) {
  const previewImage = document.getElementById("preview-image");
  previewImage.style.display = "block";
  previewImage.src = URL.createObjectURL(event.target.files[0]);
}

document.getElementById("add-user-form").addEventListener("submit", async function (event) {
  event.preventDefault(); // Ngăn chặn reload trang

  const formData = new FormData(this);

  try {
    const response = await fetch("/add_user", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    if (response.ok) {
      // Cập nhật danh sách người dùng
      const userListBody = document.getElementById("user-list-body");
      userListBody.innerHTML = ""; // Xóa danh sách cũ
      result.data.forEach((user) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${user.name}</td>
          <td>${user.image}</td>
        `;
        userListBody.appendChild(row);
      });

      alert(result.message); // Thông báo thêm thành công
    } else {
      alert(result.error); // Thông báo lỗi
    }
  } catch (error) {
    console.error("Error:", error);
    alert("An error occurred while adding the user.");
  }
});

function displayDetectedFaces(data) {
  const facesContainer = document.getElementById('faces_container');
  facesContainer.innerHTML = '';

  data.faces.forEach(face => {
    const faceDiv = document.createElement('div');
    faceDiv.className = 'face-info';

    const faceImg = document.createElement('img');
    faceImg.src = 'data:image/jpeg;base64,' + face.face_image;
    faceImg.alt = `Face`;
    faceImg.style.maxWidth = '150px';

    const nameP = document.createElement('p');
    nameP.textContent = `Name: ${face.name}`;

    const emotionP = document.createElement('p');
    emotionP.textContent = `Emotion: ${face.emotion}`;

    faceDiv.appendChild(faceImg);
    faceDiv.appendChild(nameP);
    faceDiv.appendChild(emotionP);

    facesContainer.appendChild(faceDiv);
  });
}

function displayResult(data) {
  const resultImg = document.getElementById('image_result');
  const facesContainer = document.getElementById('faces_container');

  // Hiển thị ảnh kết quả
  if (data.result_image) {
    resultImg.src = 'data:image/jpeg;base64,' + data.result_image;
  } else {
    alert('Không nhận được ảnh kết quả từ server!');
  }

  // Hiển thị thông tin khuôn mặt
  facesContainer.innerHTML = ''; // Xóa nội dung cũ
  data.faces.forEach(face => {
    const faceDiv = document.createElement('div');
    faceDiv.className = 'face-info';

    // Hiển thị ảnh khuôn mặt cắt
    const faceImg = document.createElement('img');
    faceImg.src = 'data:image/jpeg;base64,' + face.face_image;
    faceImg.alt = `Face ${face.face_index}`;
    faceImg.style.maxWidth = '150px';

    // Thông tin tên, khoảng cách và cảm xúc
    const nameP = document.createElement('p');
    nameP.textContent = `Name: ${face.name}`;

    const distanceP = document.createElement('p');
    distanceP.textContent = `Distance: ${face.distance ? face.distance.toFixed(2) : 'undefined'}`;

    const emotionP = document.createElement('p');
    emotionP.textContent = `Emotion: ${face.emotion}`;

    // Thêm thông tin vào div khuôn mặt
    faceDiv.appendChild(faceImg);
    faceDiv.appendChild(nameP);
    faceDiv.appendChild(distanceP);
    faceDiv.appendChild(emotionP);

    facesContainer.appendChild(faceDiv);
  });
}

// Gửi ảnh lên server và xử lý
fetch('/detect_embedding', {
  method: 'POST',
  body: formData,
})
  .then(response => response.json())
  .then(data => displayResult(data)) // Hiển thị ảnh và khuôn mặt
  .catch(error => console.error('Lỗi:', error));
