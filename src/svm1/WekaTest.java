/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package svm1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.GridSearch;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.TextDirectoryLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.AllFilter;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 *
 * @author hp
 */
public class WekaTest {

    public void testLoadArff() throws Exception {
        //BufferedReader reader = new BufferedReader(new FileReader("C:\\Users\\hp\\Desktop\\SVM implementation\\iris.arff"));
        //Instances data = new Instances(reader);
        DataSource source = new DataSource("C:\\Users\\hp\\Desktop\\SVM implementation\\iris.arff");
        System.out.println(source.getStructure());
        Instances data = source.getDataSet();

        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        //System.out.println(data);
        //System.out.println(data.instance(1));
        //initialize svm classifier
        LibSVM svm = new LibSVM();
        svm.buildClassifier(data);

        FastVector attributeList = new FastVector(4);
        Attribute a1 = new Attribute("sepallength");
        Attribute a2 = new Attribute("sepalwidth");
        Attribute a3 = new Attribute("petallength");
        Attribute a4 = new Attribute("petalwidth");

        FastVector classVal = new FastVector();
        classVal.addElement("Iris-setosa");
        classVal.addElement("Iris-versicolor");
        attributeList.addElement(a1);
        attributeList.addElement(a2);
        attributeList.addElement(a3);
        attributeList.addElement(a4);
        Attribute c = new Attribute("class", classVal);
        attributeList.addElement(c);

        Instances testData = new Instances("iriss", attributeList, 0);
        if (testData.classIndex() == -1) {
            testData.setClassIndex(testData.numAttributes() - 1);
        }
        Instance inst = new Instance(testData.numAttributes());
        inst.setValue(a1, 7.0);
        inst.setValue(a2, 3.2);
        inst.setValue(a3, 4.7);
        inst.setValue(a4, 1.4);
        inst.setDataset(testData);


        System.out.println(inst);
        System.out.println(testData);
        System.out.println(testData.equalHeaders(data));
        double[] v = svm.distributionForInstance(inst);
        System.out.println(v[0] + " " + v[1]);

    }

    /**
     *
     * @throws IOException
     * @throws Exception
     */
    public void loadFromTextFiles() throws IOException, Exception {

        SnowballStemmer stemmer = new SnowballStemmer();
        stemmer.setStemmer("english");
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true);
        filter.setOutputWordCounts(true);
        filter.setTFTransform(true);
        filter.setIDFTransform(true);
        filter.setStopwords(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\StopWords.txt"));
        filter.setStemmer(stemmer);
        System.out.println("%%%%%%%%%%%%%%" + filter.getStemmer());

        /* convert the directory into a dataset, text contains training date. We load them and
         then divided into two parts later*/
        TextDirectoryLoader docLoader = new TextDirectoryLoader();
        docLoader.setDirectory(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\CrimeNews"));
        Instances data = docLoader.getDataSet();
        System.out.println("\n\nImported data:\n\n" + data);

        // apply the StringToWordVector
        filter.setInputFormat(data);
        Instances dataFiltered = Filter.useFilter(data, filter);
        System.out.println("###########" + dataFiltered.numAttributes());
        System.out.println("\n\nFiltered data:\n\n" + dataFiltered);
        //System.out.println(dataFiltered.classIndex());

        LibSVM svm = new LibSVM();
        String svmOptions = "-S 0 -K 2 -C 1 -G 0.0038498";
        svm.setOptions(weka.core.Utils.splitOptions(svmOptions));
        //divide dataFiltered into training test and test set
        Instances class1Data = new Instances(dataFiltered, 0, dataFiltered.numInstances() / 2);
        Instances class2Data = new Instances(dataFiltered, dataFiltered.numInstances() / 2, dataFiltered.numInstances() / 2);
        System.out.println("\n\nClass 1 data:\n\n" + class1Data);
        System.out.println(class1Data.instance(1).classValue());
        System.out.println("\n\nClass 2 data:\n\n" + class2Data);
        System.out.println(class2Data.instance(1).classValue());

        //divide class1Data to traing and test set
        int trainSizeC1 = (int) Math.round(class1Data.numInstances() * 0.8);
        int testSizeC1 = class1Data.numInstances() - trainSizeC1;
        Instances trainC1 = new Instances(class1Data, 0, trainSizeC1);
        Instances testC1 = new Instances(class1Data, trainSizeC1, testSizeC1);
        System.out.println("\n\nClass 1  train data:\n\n" + trainC1);
        System.out.println(trainC1.instance(1).classValue());
        System.out.println("\n\nClass 1 test data:\n\n" + testC1);
        System.out.println(testC1.instance(0).classValue());

        //divide class1Data to traing and test set
        int trainSizeC2 = (int) Math.round(class2Data.numInstances() * 0.8);
        int testSizeC2 = class2Data.numInstances() - trainSizeC2;
        Instances trainC2 = new Instances(class2Data, 0, trainSizeC2);
        Instances testC2 = new Instances(class2Data, trainSizeC2, testSizeC2);
        System.out.println("\n\nClass 2  train data:\n\n" + trainC2);
        System.out.println(trainC2.instance(1).classValue());
        System.out.println("\n\nClass 2 test data:\n\n" + testC2);
        System.out.println(testC2.instance(0).classValue());


        Instances trainingSet = new Instances(trainC1);
        for (int i = 0; i < trainC2.numInstances(); i++) {
            trainingSet.add(trainC2.instance(i));
        }

        Instances testSet = new Instances(testC1);
        for (int i = 0; i < testC2.numInstances(); i++) {
            testSet.add(testC2.instance(i));
        }


        System.out.println("\n\nFinal Test:\n\n" + testSet);
        System.out.println(testSet.instance(0).classValue());
        System.out.println("\n\nFinal Training:\n\n" + trainingSet);
        System.out.println(trainingSet.instance(1).classValue());

        svm.buildClassifier(trainingSet);

        Evaluation eTest = new Evaluation(trainingSet);
        eTest.evaluateModel(svm, testSet);
        String strSummary = eTest.toSummaryString();
        System.out.println(strSummary);
        System.out.println(eTest.weightedAreaUnderROC());
    }

    /**
     *
     * @throws Exception
     */
    public void testCrossValidataion() throws Exception {

        //set tokenizer - we can specify n-grams for classification
        NGramTokenizer tokenizer = new NGramTokenizer();
        tokenizer.setNGramMinSize(1);
        tokenizer.setNGramMaxSize(1);
        tokenizer.setDelimiters("\\W");

        //set stemmer - set english stemmer
        SnowballStemmer stemmer = new SnowballStemmer();
        stemmer.setStemmer("english");

        //create new filter for vector transformation
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true);
        filter.setOutputWordCounts(true);
//      filter.setTFTransform(true);
//      filter.setIDFTransform(true);
//      filter.setStopwords(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\StopWordsR1.txt"));
        filter.setTokenizer(tokenizer);
        filter.setStemmer(stemmer);
        System.out.println("Stemmer Name- " + filter.getStemmer());

        //import data from file
        TextDirectoryLoader docLoader = new TextDirectoryLoader();
        docLoader.setDirectory(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\CrimeNews"));
        Instances data = docLoader.getDataSet();
        System.out.println("\n\nImported data:\n\n" + data);

        // apply the StringToWordVector filter
        filter.setInputFormat(data);
        Instances dataFiltered = Filter.useFilter(data, filter);
        System.out.println("Number of Attributes after stop words removal- " + dataFiltered.numAttributes());
        System.out.println("\n\nFiltered data:\n\n" + dataFiltered);

        //initialize the model and set SVM type and kernal type
        LibSVM svm = new LibSVM();
        //-S 1 -K 3 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.0010 -P 0.1 Best G 0.001953125 for C=32578
        String svmOptions = "-S 0 -K 2 -C 32578 -G 0.0038498";
        svm.setOptions(weka.core.Utils.splitOptions(svmOptions));
        System.out.println("&&&&&&&&" + svm.getSVMType() + svm.getKernelType());//1,3 best result 81%

        //select most relevant features for classification
        //dataFiltered=featureSelection(dataFiltered);
        //System.out.println("Selected Features"+dataFiltered);

        //  gridSearch(svm, dataFiltered);
        //    perform cross vlaidation
        Evaluation evaluation = new Evaluation(dataFiltered);
        evaluation.crossValidateModel(svm, dataFiltered, 2, new Random(1));
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.weightedAreaUnderROC());

        //get test instances and perform predictions
        Instances testData = createTestInstances();
        Instances testDataFiltered = Filter.useFilter(testData, filter);
        svm.buildClassifier(dataFiltered);

        for (int i = 0; i < testDataFiltered.numInstances(); i++) {
            System.out.println("\n\nTEST Filtered data:\n\n" + testDataFiltered.instance(i));
            double[] results = svm.distributionForInstance(testDataFiltered.instance(i));
            System.out.println(results[0] + " " + results[1]);
        }
        // saveClassifier(svm);
    }

    /**
     *
     * @throws Exception
     */
    public void testFilter() throws Exception {
        StringToWordVector filter = new StringToWordVector();
        String[] options = {"-C", "-T", "-I", "-S", "-L"};
        filter.setOptions(options);
        System.out.println(filter.getLowerCaseTokens());
    }

    /**
     *
     * @param svm
     * @param dataFiltered
     * @throws Exception
     */
    public void gridSearch(LibSVM svm, Instances dataFiltered) throws Exception {
        GridSearch gs = new GridSearch();
        gs.setClassifier(svm);
        gs.setFilter(new AllFilter());



        gs.setXProperty("classifier.cost");
        gs.setXMin(3);
        gs.setXMax(25);
        gs.setXStep(2);
        gs.setXBase(2);
        gs.setXExpression("pow(BASE,I)");

        gs.setYProperty("classifier.gamma");
        gs.setYMin(-8.5);
        gs.setYMax(-8);
        gs.setYStep(0.001);
        gs.setYBase(2);
        gs.setYExpression("pow(BASE,I)");
        //-y-property classifier.kernel.gamma -y-min -5.0 -y-max 2.0 -y-step 1.0 -y-base 10.0 -y-expression pow(BASE,I) -filter weka.filters.AllFilter -x-property classifier.nu -x-min 0.01 -x-max 1.0 -x-step 10.0 -x-base 10.0 -x-expression I -sample-size 100.0 -traversal COLUMN-WISE -log-file "C:\Program Files\Weka-3-6" -S 1 -W weka.classifiers.functions.LibSVM -- -S 2 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.0010 -P 0.1

        int EVALUATION_CC = 0;
        int EVALUATION_RMSE = 1;
        int EVALUATION_RRSE = 2;
        int EVALUATION_MAE = 3;
        int EVALUATION_RAE = 4;
        int EVALUATION_COMBINED = 5;
        int EVALUATION_ACC = 6;
        int EVALUATION_KAPPA = 7;
        Tag[] TAGS_EVALUATION = {
            new Tag(EVALUATION_CC, "CC", "Correlation coefficient"),
            new Tag(EVALUATION_RMSE, "RMSE", "Root mean squared error"),
            new Tag(EVALUATION_RRSE, "RRSE", "Root relative squared error"),
            new Tag(EVALUATION_MAE, "MAE", "Mean absolute error"),
            new Tag(EVALUATION_RAE, "RAE", "Root absolute error"),
            new Tag(EVALUATION_COMBINED, "COMB", "Combined = (1-abs(CC)) + RRSE + RAE"),
            new Tag(EVALUATION_ACC, "ACC", "Accuracy"),
            new Tag(EVALUATION_KAPPA, "KAP", "Kappa")
        };
        SelectedTag st = new SelectedTag(EVALUATION_ACC, TAGS_EVALUATION);
        System.out.println(st.getTags());
        gs.setEvaluation(st);

        gs.setDebug(true);

        gs.buildClassifier(dataFiltered);
        System.out.println("Criteria " + gs.getEvaluation().getSelectedTag().getID());
        System.out.println("&&&&&&&&&&&&" + gs.getValues());
    }

    /**
     *
     * @param data
     * @return
     * @throws Exception
     */
    public Instances featureSelection(Instances data) throws Exception {
        System.out.println("Attribute Selection");
        AttributeSelection filter = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);
        Instances newData = Filter.useFilter(data, filter);
        return newData;
    }

    /**
     *
     * @return @throws FileNotFoundException
     * @throws IOException
     */
    public Instances createTestInstances() throws FileNotFoundException, IOException {


        FastVector attributeList = new FastVector(2);
        Attribute a1 = new Attribute("text", (FastVector) null);

        FastVector classVal = new FastVector();
        classVal.addElement("crimes");
        classVal.addElement("others");
        Attribute c = new Attribute("@@class@@", classVal);

        //add class attribute and news text
        attributeList.addElement(a1);
        attributeList.addElement(c);



        Instances testData = new Instances("TestNews", attributeList, 0);
        if (testData.classIndex() == -1) {
            testData.setClassIndex(1);
        }
        //for each test instance
        File newsDirectory = new File("C:\\Users\\hp\\Desktop\\SVM implementation\\TestNews");
        BufferedReader newsArticleReader;
        for (File fileEntry : newsDirectory.listFiles()) {
            System.out.println(fileEntry.getPath());
            newsArticleReader = new BufferedReader(new FileReader(fileEntry.getPath()));
            String news = "";
            String newsLine = "";
            while ((newsLine = newsArticleReader.readLine()) != null) {
                news = news.concat(newsLine);
                news = news.concat(" ");
            }
            Instance inst = new Instance(testData.numAttributes());
            inst.setValue(a1, news);

            inst.setDataset(testData);
            inst.setClassMissing();

            System.out.println(inst);
            testData.add(inst);
        }
        return testData;
    }

    private void saveClassifier(LibSVM svm) throws IOException {
        //save instance
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("/some/where/j48.model"));
        oos.writeObject(svm);
        oos.flush();
        oos.close();
    }

    public void testMultiClassClassification() throws Exception {
        //set tokenizer - we can specify n-grams for classification
        NGramTokenizer tokenizer = new NGramTokenizer();
        tokenizer.setNGramMinSize(1);
        tokenizer.setNGramMaxSize(1);
        tokenizer.setDelimiters("\\W");

        //set stemmer - set english stemmer
        SnowballStemmer stemmer = new SnowballStemmer();
        stemmer.setStemmer("english");

        //create new filter for vector transformation
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true);
        filter.setOutputWordCounts(true);
//      filter.setTFTransform(true);
//      filter.setIDFTransform(true);
//      filter.setStopwords(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\StopWordsR1.txt"));
        filter.setTokenizer(tokenizer);
        filter.setStemmer(stemmer);
        System.out.println("Stemmer Name- " + filter.getStemmer());

        //import data from file
        TextDirectoryLoader docLoader = new TextDirectoryLoader();
        docLoader.setDirectory(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\CrimeAccidentsOthers"));
        Instances data = docLoader.getDataSet();
        System.out.println("\n\nImported data:\n\n" + data);

        // apply the StringToWordVector filter
        filter.setInputFormat(data);
        Instances dataFiltered = Filter.useFilter(data, filter);
        System.out.println("Number of Attributes after stop words removal- " + dataFiltered.numAttributes());
        System.out.println("\n\nFiltered data:\n\n" + dataFiltered);

        //initialize the model and set SVM type and kernal type
        LibSVM svm = new LibSVM();
        //-S 1 -K 3 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.0010 -P 0.1 Best G 0.001953125 for C=32578
        String svmOptions = "-S 0 -K 2 -C 32578 -G 0.0038498";
        svm.setOptions(weka.core.Utils.splitOptions(svmOptions));
        System.out.println("&&&&&&&&" + svm.getSVMType() + svm.getKernelType());//1,3 best result 81%

        //select most relevant features for classification
        //dataFiltered=featureSelection(dataFiltered);
        //System.out.println("Selected Features"+dataFiltered);

        //  gridSearch(svm, dataFiltered);
        //    perform cross vlaidation
        Evaluation evaluation = new Evaluation(dataFiltered);
        evaluation.crossValidateModel(svm, dataFiltered, 2, new Random(1));
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.weightedAreaUnderROC());

       

    }
    
    /**
     * this is implemented as per given in http://jmgomezhidalgo.blogspot.com.es/2013/01/text-mining-in-weka-chaining-filters.html
     * reason for using this when classifying instances in real world what we are doing is first use filter
     * to transform training data and then use the same filter for test data. When we use filtered classifer
     * same thing happens. but if we use previous approach then first we transform all the data (training and test)
     * and then give it to the classifier. It is not similar to real world scenario.
     */
    public void testChainClassifier() throws IOException, Exception{
        //set tokenizer - we can specify n-grams for classification
        NGramTokenizer tokenizer = new NGramTokenizer();
        tokenizer.setNGramMinSize(1);
        tokenizer.setNGramMaxSize(1);
        tokenizer.setDelimiters("\\W");

        //set stemmer - set english stemmer
        SnowballStemmer stemmer = new SnowballStemmer();
        stemmer.setStemmer("english");

        //create new filter for vector transformation
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true);
        filter.setOutputWordCounts(true);
//      filter.setTFTransform(true);
//      filter.setIDFTransform(true);
//      filter.setStopwords(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\StopWordsR1.txt"));
        filter.setTokenizer(tokenizer);
        filter.setStemmer(stemmer);
        System.out.println("Stemmer Name- " + filter.getStemmer());

        //import data from file
        TextDirectoryLoader docLoader = new TextDirectoryLoader();
        docLoader.setDirectory(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\CrimeNews"));
        Instances data = docLoader.getDataSet();
        System.out.println("\n\nImported data:\n\n" + data);

        //create a filtered classifier
        FilteredClassifier fc=new FilteredClassifier();
        
        //initialize the model and set SVM type and kernal type
        LibSVM svm = new LibSVM();
        //-S 1 -K 3 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.0010 -P 0.1 Best G 0.001953125 for C=32578
        String svmOptions = "-S 0 -K 2 -C 32578 -G 0.0038498";
        svm.setOptions(weka.core.Utils.splitOptions(svmOptions));
        System.out.println("&&&&&&&&" + svm.getSVMType() + svm.getKernelType());//1,3 best result 81%

        // set classifier and filter for filtered classifier
        fc.setClassifier(svm);
        fc.setFilter(filter);
        
        // perform cross vlaidation
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(fc, data, 2, new Random(1));
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.weightedAreaUnderROC());
    }
}
