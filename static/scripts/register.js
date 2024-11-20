const form = document.getElementById("registerForm");
const emailField = document.getElementById("email");
const passwordField = document.getElementById("password");
const confirmPasswordField = document.getElementById("confirm_password");
const errorMessage = document.getElementById("error_message");

function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/; // Biểu thức regex kiểm tra email
  return emailRegex.test(email);
}

function validateForm(event) {
  const email = emailField.value;
  const password = passwordField.getAttribute("data-real-value");
  const confirmPassword = confirmPasswordField.getAttribute("data-real-value");

  if (!validateEmail(email)) {
    event.preventDefault(); 
    errorMessage.textContent = "Invalid email format!";
    return;
  }

  if (password !== confirmPassword) {
    event.preventDefault();
    errorMessage.textContent = "Passwords do not match!";
    return;
  }

  errorMessage.textContent = "";
  passwordField.value = password;
  confirmPasswordField.value = confirmPassword;
}

function maskPassword(inputField) {
  const realValue = inputField.value; 
  const maskedValue = "*".repeat(realValue.length); 
  inputField.setAttribute("data-real-value", realValue); 
  inputField.value = maskedValue; 
}

passwordField.addEventListener("input", () => maskPassword(passwordField));
confirmPasswordField.addEventListener("input", () =>
  maskPassword(confirmPasswordField)
);

form.addEventListener("submit", validateForm);
