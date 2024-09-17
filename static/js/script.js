document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('file-input');
    const resultDiv = document.getElementById('result');
    const uploadedImage = document.getElementById('uploaded-image');
    const predictionOutput = document.getElementById('prediction-output');

    // Check if a file was selected
    if (fileInput.files && fileInput.files[0]) {
        // Create a FormData object and append the file
        let formData = new FormData();
        formData.append('file', fileInput.files[0]);

        // Display the uploaded image before sending it to the server
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImage.src = e.target.result;
            uploadedImage.style.display = 'block';
        };
        reader.readAsDataURL(fileInput.files[0]);

        // Send the image file to the server using fetch
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Check for errors
            if (data.error) {
                predictionOutput.textContent = `Error: ${data.error}`;
            } else {
                // Display the processed image with bounding boxes
                uploadedImage.src = 'data:image/png;base64,' + data.image;
                uploadedImage.style.display = 'block';

                // Clear previous predictions
                predictionOutput.innerHTML = '';

                // Display all predictions
                data.predictions.forEach(pred => {
                    const predictionText = `Gender: ${pred.gender}, Age: ${pred.age}`;
                    const p = document.createElement('p');
                    p.textContent = predictionText;
                    predictionOutput.appendChild(p);
                });
            }
        })
        .catch(error => {
            console.error('Error:', error);
            predictionOutput.textContent = 'An error occurred. Please try again.';
        });
    } else {
        predictionOutput.textContent = 'Please select an image file.';
    }
});

// Handle webcam functionality
const cameraBtn = document.getElementById('camera-btn');
const webcam = document.getElementById('webcam');
const predictWebcamBtn = document.getElementById('predict-webcam-btn');
const stopWebcamBtn = document.getElementById('stop-webcam-btn');
const predictionOutput = document.getElementById('prediction-output');

let webcamStream;

// Open webcam
cameraBtn.addEventListener('click', function() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                webcamStream = stream;
                webcam.srcObject = stream;
                webcam.style.display = 'block';
                predictWebcamBtn.style.display = 'block';
                stopWebcamBtn.style.display = 'block';  // Show stop button when camera is open
                
                // Set canvas size to match the video size
                overlayCanvas.width = webcam.videoWidth;
                overlayCanvas.height = webcam.videoHeight;
                overlayCanvas.style.display = 'block';
            })
            .catch(function(err) {
                console.error("Error accessing webcam: ", err);
            });
    }
});

// Stop webcam
stopWebcamBtn.addEventListener('click', function() {
    if (webcamStream) {
        const tracks = webcamStream.getTracks();
        tracks.forEach(track => track.stop());  // Stop each track in the stream
        webcam.style.display = 'none';          // Hide the webcam element
        predictWebcamBtn.style.display = 'none'; // Hide the predict button
        stopWebcamBtn.style.display = 'none';    // Hide the stop button
        overlayCanvas.style.display = 'none';    // Hide the overlay canvas
        webcamStream = null;                    // Clear the stream variable
    }
});

// Predict with webcam
predictWebcamBtn.addEventListener('click', function() {
    const canvas = document.createElement('canvas');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    canvas.getContext('2d').drawImage(webcam, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(function(blob) {
        const formData = new FormData();
        formData.append('file', blob, 'webcam-image.png');

        // Send the webcam image to the server for prediction
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                predictionOutput.textContent = `Error: ${data.error}`;
            } else {
                predictionOutput.innerHTML = '';
                data.predictions.forEach(pred => {
                    const predictionText = `Gender: ${pred.gender}, Age: ${pred.age}`;
                    const p = document.createElement('p');
                    p.textContent = predictionText;
                    predictionOutput.appendChild(p);
                });
            }
        })
        .catch(error => {
            console.error('Error:', error);
            predictionOutput.textContent = 'An error occurred. Please try again.';
        });
    });
});