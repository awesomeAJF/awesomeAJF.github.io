const predResultElement = document.getElementById('predResult')

$("#img_input").on("change", function(e){

  var file = e.target.files[0]; //获取图片资源

  // 只选择图片文件
  if (!file.type.match('image.*')) {
    return false;
  }

  var reader = new FileReader();

  reader.readAsDataURL(file); // 读取文件

  // 渲染文件
  reader.onload = function(arg) {

    var img = '<img class="preview" src="' + arg.target.result + '" alt="preview"/>';
    $(".preview_box").empty().append(img);

    const image = new Image();
    image.onload = function() {
      raw_input = tf.browser.fromPixels(image)
      raw_input.print() // get image tensor

      // init model
      // const model = createModel('./model.json')
      let model;
      model = createConvModel()

      // preprocess
      var input = preprocessImage(raw_input)

      // predict
      console.log(input.shape)
      console.log(input.array())
      const result = model.predict(raw_input.as4D(1, 28, 28, 3))
      const pred = Array.from(result.argMax(1).dataSync());

      // print result
      showPredictionResult(pred)

    }
    image.src=arg.target.result
  }
});

function preprocessImage(raw_image) {

  var img = tf.expandDims(raw_image)
  return img

}

function createModel(url) {
  console.log(`creating model from: ${url}`);
  const model = tf.loadLayersModel(`${url}`);
  model.summary();
  return model;
}

function createConvModel() {
  // Create a sequential neural network model. tf.sequential provides an API
  // for creating "stacked" models where the output from one layer is used as
  // the input to the next layer.
  const model = tf.sequential();

  // The first layer of the convolutional neural network plays a dual role:
  // it is both the input layer of the neural network and a layer that performs
  // the first convolution operation on the input. It receives the 28x28 pixels
  // black and white images. This input layer uses 16 filters with a kernel size
  // of 5 pixels each. It uses a simple RELU activation function which pretty
  // much just looks like this: __/
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 3],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));

  // After the first layer we include a MaxPooling layer. This acts as a sort of
  // downsampling using max values in a region instead of averaging.
  // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

  // Our third layer is another convolution, this time with 32 filters.
  model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));

  // Max pooling again.
  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

  // Add another conv2d layer.
  model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({units: 64, activation: 'relu'}));

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
  // represent numbers, but it's the same idea if you had classes that
  // represented other entities like dogs and cats (two output classes: 0, 1).
  // We use the softmax function as the activation for the output layer as it
  // creates a probability distribution over our 10 classes so their output
  // values sum to 1.
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  return model;
}

function showPredictionResult(prediction) {

  predResultElement.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'pred-container';

  const pred = document.createElement('div');
  pred.className = `prediction`;
  pred.innerText = `prediction: ${prediction}`;
  div.appendChild(pred);

  predResultElement.appendChild(div);
}
