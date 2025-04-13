import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const drawUtils = new DrawingUtils(canvasCtx);
let handLandmarker;
let webcamRunning = false;
let handResults;

const testResultsDiv = document.getElementById('testResults');
const hiddenContainerUntilTrained = document.getElementById('hiddenContainerUntilTrained');
hiddenContainerUntilTrained.style.display = "none";
const inputField = document.getElementById('inputField');
const sumWithAnswer = document.getElementById('sumWithAnswer');

const calculatorBackspace = document.getElementById('calculatorBackspace');
const calculatorClear = document.getElementById('calculatorClear');
const calculatorCalculate = document.getElementById('calculatorCalculate');

calculatorBackspace.addEventListener("click", calculatorBackspaceFunction);
calculatorClear.addEventListener("click", calculatorClearFunction);
calculatorCalculate.addEventListener("click", calculatorCalculateFunction);

let currentSum = []
let nn

function createNeuralNetwork() {
    ml5.setBackend("webgl")
    nn = ml5.neuralNetwork({ task: 'classification', debug: true })

    const options = {
        model: "./model/model.json",
        metadata: "./model/model_meta.json",
        weights: "./model/model.weights.bin"
    }
    console.log(options)
    nn.load(options, finishedTraining)
    console.log(nn)
}

function finishedTraining() {
    console.log("finished training")
    hiddenContainerUntilTrained.style.display = "";
}

async function calculateLetter(letterArray) {
    nn.classify(letterArray, (results) => {
        console.log(results)
        testResultsDiv.textContent = `I think this is ${results[0].label}, with a ${results[0].confidence.toFixed(2) * 100}% accuracy`
        currentSum.push(results[0].label)
        inputField.innerHTML = currentSum.join("")
    })
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

    console.log("All models loaded. You can start the webcam.");
    enableWebcamButton.addEventListener("click", toggleWebcam);
    logButton.addEventListener("click", logResults);
};

// Toggle webcam
async function toggleWebcam() {
    if (webcamRunning) {
        webcamRunning = false;
        enableWebcamButton.innerText = "Start tracking";
        return;
    }

    webcamRunning = true;
    enableWebcamButton.innerText = "Stop tracking";

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
    if (!handLandmarker) return;

    handResults = await handLandmarker.detectForVideo(video, performance.now());

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
}

// Log detected landmarks
function logResults() {
    console.log("Hand landmarks:", handResults?.landmarks);
    logToData()
}

// Start the application
if (navigator.mediaDevices?.getUserMedia) {
    createNeuralNetwork();
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

function calculatorBackspaceFunction() {
    const removedCharacter = currentSum.pop()
    console.log(removedCharacter)
    inputField.innerHTML = currentSum.join("")
}
function calculatorClearFunction() {
    console.log(currentSum)
    currentSum = []
    inputField.innerHTML = currentSum.join("")
}
function calculatorCalculateFunction() {
    if (currentSum.length === 0) {
        displaySumFunction('Add a pose to the sum first');
    } else {
        const result = calculateFullSum();
        console.log(result);
        displaySumFunction(result);
    }
}

function calculateFullSum() {
    //met behulp van ai het rekenmachinesysteem gemaakt
    const prettyExpression = currentSum.map(char => {
        if (char === 'x') return ' * ';
        if (['+', '-', '/'].includes(char)) return ` ${char} `;
        return char;
    }).join('');

    const evalExpression = currentSum.map(char => {
        return char === 'x' ? '*' : char;
    }).join('');

    try {
        const result = eval(evalExpression);
        return `${prettyExpression} = ${result}`;
    } catch (error) {
        console.error('Error evaluating expression:', error);
        return null;
    }
}

function displaySumFunction(result) {
    sumWithAnswer.innerHTML = result
}
