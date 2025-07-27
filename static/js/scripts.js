function predictHeartDisease() {
  // Use the correct API URL for heart disease prediction
  handlePrediction("heart-disease-form", "/predict/heart", "result-heart");
}

function predictLiverDisease() {
  // Use the correct API URL for liver disease prediction
  handlePrediction("liver-disease-form", "/predict/liver", "result-liver");
}
function predictKidneyDisease() {
  // Use the correct API URL for liver disease prediction
  handlePrediction("kidney-disease-form", "/predict/kidney", "result-kidney");
}
function predictDiabetes() {
  // Use the correct API URL for liver disease prediction
  handlePrediction("diabetes-form", "/predict/diabetes", "result-diabetes");
}
function handlePrediction(formId, apiUrl, resultId) {
  const form = document.getElementById(formId);
  const formData = new FormData(form);
  const data = Object.fromEntries(formData.entries());

  fetch(apiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then((result) => {
      document.getElementById(resultId).textContent = JSON.stringify(
        result,
        null,
        2
      );
    })
    .catch((error) => {
      document.getElementById(resultId).textContent =
        "Prediction failed: " + error.message;
      console.error("Error:", error);
    });
}


