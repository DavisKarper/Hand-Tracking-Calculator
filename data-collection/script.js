import { HandLandmarker, PoseLandmarker, FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const drawUtils = new DrawingUtils(canvasCtx);
let handLandmarker, poseLandmarker, faceLandmarker;
let webcamRunning = false;
let handResults, poseResults, faceResults;

const poseLabel = document.getElementById("poseLabel")
const showAllDataButton = document.getElementById("showAllDataButton")
let allLetterArrays = []
showAllDataButton.onclick = () => {
    console.log(allLetterArrays)
}

async function calculateLetter(letterArray) {
    allLetterArrays.push({
        data: letterArray,
        label: poseLabel.value
    });
    console.log(allLetterArrays)
}

// Initialize all trackers
const initializeTrackers = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 2
    });

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1
    });

    console.log("All models loaded. You can start the webcam.");
    enableWebcamButton.addEventListener("click", toggleWebcam);
    logButton.addEventListener("click", logResults);
};

// Toggle webcam
async function toggleWebcam() {
    if (webcamRunning) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE TRACKING";
        return;
    }

    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE TRACKING";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

// Run predictions for all models
async function predictWebcam() {
    if (!handLandmarker || !poseLandmarker || !faceLandmarker) return;

    handResults = await handLandmarker.detectForVideo(video, performance.now());
    poseResults = await poseLandmarker.detectForVideo(video, performance.now());
    faceResults = await faceLandmarker.detectForVideo(video, performance.now());

    drawResults();
    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// Draw results on the canvas
function drawResults() {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Draw hand landmarks
    if (handResults?.landmarks) {
        for (let hand of handResults.landmarks) {
            drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
            drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
        }
    }

    // Draw pose landmarks
    if (poseResults?.landmarks) {
        for (const landmark of poseResults.landmarks) {
            drawUtils.drawLandmarks(landmark, { radius: 3 });
            drawUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        }
    }

    // Draw face landmarks
    if (faceResults?.faceLandmarks) {
        for (const landmarks of faceResults.faceLandmarks) {
            drawUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
            drawUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
            drawUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
        }
    }
}

// Log detected landmarks
function logResults() {
    console.log("Hand landmarks:", handResults?.landmarks);
    console.log("Pose landmarks:", poseResults?.landmarks);
    console.log("Face landmarks:", faceResults?.faceLandmarks);
    logToData()
}

// Start the application
if (navigator.mediaDevices?.getUserMedia) {
    initializeTrackers();
}

function logToData() {
    let newLetterArray = []
    let handArray = handResults?.landmarks
    handArray[0].forEach(point => {
        newLetterArray = [...newLetterArray, point["x"], point["y"], point["z"]]
    });
    console.log(newLetterArray)
    calculateLetter(newLetterArray)
}

