package net.zomis.sweeplearn;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.CropImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.imgscalr.Scalr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.IntStream;

public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    protected static int height = 90;
    protected static int width = 90;
    protected static int channels = 3;
    protected static int numExamples = 80;
    protected static int numLabels = 3;
    protected static int batchSize = 20;

    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 100;
    protected static double splitTrainTest = 0.8;
    protected static boolean save = false;
    private final NativeImageLoader imgLoader = new NativeImageLoader(height, width, channels);
    private MultiLayerNetwork network;

    public static void main(String[] args) {
        new Main().run();
    }

    private void run() {
        network = alexnetModel();
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        String mypath = "src/main/resources/sweepgrid/";
        logger.info(System.getProperty("user.dir") + "======" + mypath);
        File mainPath = new File(System.getProperty("user.dir"), mypath);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

        /**
         * Data Setup -> train test split
         *  - inputSplit = define train and test split
         **/
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
//        InputSplit testData = inputSplit[1];
//        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        network.init();

//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);
//        network.setListeners((IterationListener)new StatsListener( statsStorage),new ScoreIterationListener(iterations));
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;

        try (ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker)) {
            recordReader.initialize(trainData, null);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
//            scaler.fit(dataIter);
//            dataIter.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, dataIter);
            logger.info("Network fitting...");
            network.fit(trainIter);
            logger.info("Network fit!");

        } catch (IOException e) {
            e.printStackTrace();
        }

        int bigImageWidth = 1920;
        int bigImageHeight = 1080;
        int x = 628;
        int y = 110;
        int right = bigImageWidth - x - width;
        int bottom = bigImageHeight - y - height;
        logger.info("right {}, bottom {}", right, bottom);
        CropImageTransform imgCrop = new CropImageTransform(y, x, bottom, right);
        //new RandomCropTransform()

//        NativeImageLoader imgLoader = new NativeImageLoader(bigImageHeight, bigImageWidth, channels, imgCrop);

        play(() -> getClass().getClassLoader().getResourceAsStream("9x9-4.png"));
    }

    private void play(Supplier<InputStream> resource) {
        int left = 628 - 96;
        int top = 110;
        int xx = left;
        int yy = top;
        while (xx < 1350) {
            double[] results = imageAt(resource.get(), xx, yy);
            logger.info("Results at {}, {} = {}", xx, yy, Arrays.toString(results));
            xx += 96;
        }
    }

    private double[] imageAt(InputStream resource, int x, int y) {
        try {
            logger.info("Image at {}, {}", x, y);
            // InputStream resource = new FileInputStream("image.png");
            InputStream stream = croppedStream(resource, x, y);
            logger.info("Loading matrix");
            INDArray inputMatrix = imgLoader.asMatrix(stream);
            logger.info("Activating network with input with Matrix size " + Arrays.toString(inputMatrix.shape()));
            INDArray result = network.output(inputMatrix, false);// activate();// activate(inputMatrix);
            return IntStream.range(0, result.size(1)).mapToDouble(i -> result.getDouble(0, i)).toArray();
        } catch (IOException e) {
            logger.error(e.toString(), e);
            return new double[numLabels];
        }
    }

    private InputStream croppedStream(InputStream imageInput, int x, int y) throws IOException {
        BufferedImage img = ImageIO.read(imageInput);
        img = Scalr.crop(img, x, y, width, height);
        ByteArrayOutputStream tempStream = new ByteArrayOutputStream();
        if (true) {
            File testFile = new File("temp" + x + "__" + y + ".png");
            ImageIO.write(img, "png", testFile);
        }
        ImageIO.write(img, "png", tempStream);
        byte[] output = tempStream.toByteArray();
        return new ByteArrayInputStream(output);
    }

    public MultiLayerNetwork alexnetModel() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(0.9))
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2)
                .biasLearningRate(1e-2*2)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .list()
                .layer(0, convInit("cnn1", channels, 24, new int[]{5, 5}, new int[]{2, 2}, new int[]{3, 3}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[] {2,2}))
                .layer(3, conv5x5("cnn2", 24, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[]{3,3}))
                .layer(6,conv3x3("cnn3", 24, 0))
//                .layer(7,conv3x3("cnn4", 384, nonZeroBias))
//                .layer(8,conv3x3("cnn5", 256, nonZeroBias))
                .layer(7, maxPool("maxpool3", new int[]{3,3}))
                .layer(8, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
//                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

}
