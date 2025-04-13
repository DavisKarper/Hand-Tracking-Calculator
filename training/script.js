import posedata from './data.json' with {type: "json"};

const statusDiv = document.getElementById('status');
const downloadButton = document.getElementById('downloadButton');
downloadButton.style.display = "none";
let nn

function startTraining() {
    statusDiv.textContent = 'Status: Training neural network ... '

    ml5.setBackend("webgl")
    const options = {
        task: 'classification',
        debug: true,
        learningRate: 0.2,
        layers: [
            {
                type: 'dense',
                units: 32,
                activation: 'relu',
            }, {
                type: 'dense',
                units: 16,
                activation: 'relu',
            }, {
                type: 'dense',
                activation: 'softmax',
            },
        ]
    }
    nn = ml5.neuralNetwork(options)

    console.log(nn)
    posedata.forEach(pose => {
        nn.addData(pose.data, { label: pose.label })
    });

    nn.normalizeData()

    nn.train({ epochs: 30, batchSize: 12 }, finishedTraining)
}

function finishedTraining() {
    console.log("finished training")
    statusDiv.textContent = 'Status: Training done!'
    downloadButton.style.display = "";
    downloadButton.addEventListener('click', () => {
        nn.save();
    });
}

startTraining()