const inputImageElement = document.getElementById('inputImage')
const predResultElement = document.getElementById('predResult')
const fileInput = document.getElementById('test-image-file')
const info = document.getElementById('test-file-info')
const preview = document.getElementById('test-image-preview')
var raw_input

// 监听change事件:
fileInput.addEventListener('change', function () {
    // 清除背景图片:
    preview.style.backgroundImage = '';
    // 检查文件是否选择:
    if (!fileInput.value) {
        info.innerHTML = '没有选择文件';
        return;
    }
    // 获取File引用:
    var file = fileInput.files[0];
    if (file.type !== 'image/jpeg' && file.type !== 'image/png' && file.type !== 'image/gif') {
        alert('不是有效的图片文件!');
        return;
    }
    // 读取文件:
    var reader = new FileReader();
    reader.onload = function(e) {
      var data = e.target.result; // 'data:image/jpeg;base64,/9j/4AAQSk...(base64编码)...'            
      preview.style.backgroundImage = 'url(' + data + ')';

      const image = new Image();
      image.onload = function() {
        raw_input = tf.browser.fromPixels(image)
        raw_input.print()

        showImagePredictionResult(raw_input, 2)

      }
      image.src=data
    };
    reader.readAsDataURL(file);
    // 以DataURL的形式读取文件:
});

// draw image to canvas

function showImagePredictionResult(image, prediction) {

  predResultElement.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'pred-container';

  const canvas = document.createElement('canvas');
  canvas.className = 'prediction-canvas';
  draw(image.flatten(), canvas, image.shape[0], image.shape[1]);

  const pred = document.createElement('div');
  pred.className = `prediction`;
  pred.innerText = `prediction: ${prediction}`;
  div.appendChild(pred);
  div.appendChild(canvas);

  predResultElement.appendChild(div);
}

function draw(image, canvas, height, width) {
  // const [width, height] = [height, width];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function createModel(model_path) {
  console.log(`creating model from: ${model_path}`);
  const model = tf.loadLayersModel(`${model_path}`);
  model.summary();
  return model;
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

// function setPredictButtonCallback(callback) {
//   const predictButton = document.getElementById('start');
//   predictButton.addEventListener('click', () => {
//     predictButton.setAttribute('disabled', true);
//     callback();
//   });
// }

// setPredictButtonCallback(async () => {
//   // logStatus('Loading MNIST data...');
//   await load();

//   // logStatus('Creating model...');
//   const model = createModel('./model.json');
//   model.summary();

//   // logStatus('Starting model training...');
//   // await train(model, () => showPredictions(model));
//   train(model);
// });
