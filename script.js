// Analyze Song Function
async function analyzeSong() {
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = ''; // Clear previous results

    // Check if the user has selected a file
    if (!fileInput.files.length) {
        resultDiv.innerHTML = '<p class="error">Please select a file first.</p>';
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        // Sending the file to the backend for genre prediction
        const response = await fetch('http://localhost:5000/analyze', { // Backend endpoint for analysis
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            // Display predicted genre
            resultDiv.innerHTML = `<p>Predicted Genre: <strong>${data.genre}</strong></p>`;

            // Add feedback options (correct/incorrect)
            resultDiv.innerHTML += `
                <p>Was this prediction correct?</p>
                <button onclick="sendFeedback(true, '${data.genre}')">Yes</button>
                <button onclick="sendFeedback(false, '${data.genre}')">No</button>
            `;
        } else {
            // Handle error if something went wrong
            resultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
        }
    } catch (error) {
        resultDiv.innerHTML = '<p class="error">Failed to connect to the server.</p>';
    }
}

// Send feedback (correct/incorrect)
async function sendFeedback(isCorrect, predictedGenre) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '<p>Sending feedback...</p>';

    const feedbackData = {
        correct: isCorrect,
        genre: predictedGenre
    };

    try {
        const response = await fetch('http://localhost:5000/feedback', { // Backend endpoint for feedback
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedbackData)
        });

        const data = await response.json();

        if (response.ok) {
            resultDiv.innerHTML = `<p>Feedback submitted: ${isCorrect ? 'Correct' : 'Incorrect'}</p>`;
        } else {
            resultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
        }
    } catch (error) {
        resultDiv.innerHTML = '<p class="error">Failed to send feedback.</p>';
    }
}

// Event listener for triggering the analyze function when a file is selected
document.getElementById('fileInput').addEventListener('change', analyzeSong);
