// Select các phần tử cần thiết
const form = document.getElementById("loginForm");
const usernameField = document.getElementById("username");
const passwordField = document.getElementById("password");
const showPasswordCheckbox = document.getElementById("show_password");
const errorMessage = document.getElementById("error_message");

// Hàm kiểm tra không để trống khi submit
form.addEventListener("submit", (e) => {
  const username = usernameField.value.trim();
  const password = passwordField.value.trim();

  if (!username || !password) {
    e.preventDefault(); // Ngăn không cho form submit
    errorMessage.textContent = "Username and Password cannot be empty!";
    return;
  }

  errorMessage.textContent = ""; // Xóa thông báo lỗi nếu không có lỗi
});

showPasswordCheckbox.addEventListener("change", () => {
  passwordField.type = showPasswordCheckbox.checked ? "text" : "password";
});
