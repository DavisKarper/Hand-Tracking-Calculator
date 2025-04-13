import testdata from './testdata.json' with {type: "json"};

const statusDiv = document.getElementById('status');
const testResultsDiv = document.getElementById('testResults');
let nn

function createNeuralNetwork() {
    statusDiv.textContent = 'Status: Training neural network ... '

    ml5.setBackend("webgl")
    nn = ml5.neuralNetwork({ task: 'classification', debug: true })

    const options = {
        model: "../model/model.json",
        metadata: "../model/model_meta.json",
        weights: "../model/model.weights.bin"
    }
    console.log(options)
    nn.load(options, finishedTraining)
    console.log(nn)
}

function finishedTraining() {
    console.log("finished training")
    statusDiv.textContent = 'Status: Training done!'
    calculateLetter()
}

async function calculateLetter() {
    let c = 0;
    let total = testdata.length;

    for (const pose of testdata) {
        const poseArray = pose.data;
        const result = await classifyAsync(poseArray);
        console.log(result.label, pose.label);
        if (result.label === pose.label) {
            c += 1;
        }
    }

    showInBrowser(total, c);
}

function showInBrowser(i, c) {
    testResultsDiv.textContent = `${c}/${i} correct answers. Accuracy is ${(c / i * 100).toFixed(2)}%`;
}

// Helper functie om callback naar Promise om te zetten
function classifyAsync(input) {
    return new Promise((resolve, reject) => {
        nn.classify(input, (results) => {
            if (results && results[0]) {
                resolve(results[0]);
            } else {
                reject(new Error("No results"));
            }
        });
    });
}

// Start the application
if (navigator.mediaDevices?.getUserMedia) {
    createNeuralNetwork();
}