/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package svm1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.SparseInstance;
import weka.core.Tag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.TextDirectoryLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
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

    public void loadFromTextFiles() throws IOException, Exception {

        SnowballStemmer stemmer = new SnowballStemmer();
        stemmer.setStemmer("english");
        StringToWordVector filter = new StringToWordVector();
        String[] options = {"-C","-T","-I","-L"};
        filter.setOptions(options);
        filter.setStopwords(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\StopWords.txt"));
        filter.setStemmer(stemmer);
        System.out.println("%%%%%%%%%%%%%%"+filter.getStemmer());

        /* convert the directory into a dataset, text contains training date. We load them and
        then divided into two parts later*/
        
        TextDirectoryLoader docLoader = new TextDirectoryLoader();
        docLoader.setDirectory(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\News"));
        Instances data = docLoader.getDataSet();
        System.out.println("\n\nImported data:\n\n" + data);

        // apply the StringToWordVector
        filter.setInputFormat(data);
        Instances dataFiltered = Filter.useFilter(data, filter);
        System.out.println("###########"+dataFiltered.numAttributes());
        System.out.println("\n\nFiltered data:\n\n" + dataFiltered);
        //System.out.println(dataFiltered.classIndex());

        LibSVM svm = new LibSVM();
        String svmOptions = "-S 1 -K 3";
        svm.setOptions(weka.core.Utils.splitOptions(svmOptions));
        //divide dataFiltered into training test and test set
        Instances class1Data = new Instances(dataFiltered, 0, dataFiltered.numInstances() / 2);
        Instances class2Data = new Instances(dataFiltered, dataFiltered.numInstances() / 2, dataFiltered.numInstances() / 2);
        System.out.println("\n\nClass 1 data:\n\n" +class1Data);
        System.out.println(class1Data.instance(1).classValue());
        System.out.println("\n\nClass 2 data:\n\n" +class2Data);
        System.out.println(class2Data.instance(1).classValue());

        //divide class1Data to traing and test set
        int trainSizeC1 = (int) Math.round(class1Data.numInstances() * 0.8);
        int testSizeC1 = class1Data.numInstances() - trainSizeC1;
        Instances trainC1 = new Instances(class1Data, 0, trainSizeC1);
        Instances testC1 = new Instances(class1Data, trainSizeC1, testSizeC1);
        System.out.println("\n\nClass 1  train data:\n\n" +trainC1);
        System.out.println(trainC1.instance(1).classValue());
        System.out.println("\n\nClass 1 test data:\n\n" +testC1);
        System.out.println(testC1.instance(0).classValue());
        
        //divide class1Data to traing and test set
        int trainSizeC2 = (int) Math.round(class2Data.numInstances() * 0.8);
        int testSizeC2 = class2Data.numInstances() - trainSizeC2;
        Instances trainC2= new Instances(class2Data, 0, trainSizeC2);
        Instances testC2 = new Instances(class2Data, trainSizeC2, testSizeC2);
        System.out.println("\n\nClass 2  train data:\n\n" +trainC2);
        System.out.println(trainC2.instance(1).classValue());
        System.out.println("\n\nClass 2 test data:\n\n" +testC2);
        System.out.println(testC2.instance(0).classValue());
        
        
        Instances trainingSet=new Instances(trainC1);
        for(int i=0;i<trainC2.numInstances();i++){
            trainingSet.add(trainC2.instance(i));
        }
        
        Instances testSet=new Instances(testC1);
        for(int i=0;i<testC2.numInstances();i++){
            testSet.add(testC2.instance(i));
        }
        
        
        System.out.println("\n\nFinal Test:\n\n" +testSet);
        System.out.println(testSet.instance(0).classValue());
        System.out.println("\n\nFinal Training:\n\n" +trainingSet);
        System.out.println(trainingSet.instance(1).classValue());
        
        svm.buildClassifier(trainingSet);
        
        Evaluation eTest = new Evaluation(trainingSet);
        eTest.evaluateModel(svm, testSet);
        String strSummary = eTest.toSummaryString();
        System.out.println(strSummary);
        System.out.println(eTest.weightedAreaUnderROC());
    }
    
    public void testCrossValidataion() throws Exception{
        
        //set tokenizer - we can specify n-grams for classification
        NGramTokenizer tokenizer = new NGramTokenizer();
        tokenizer.setNGramMinSize(3);
        tokenizer.setNGramMaxSize(3);
        tokenizer.setDelimiters("\\W");

        //set stemmer - set english stemmer
        SnowballStemmer stemmer = new SnowballStemmer();
        stemmer.setStemmer("english");
        
        //create new filter for vector transformation
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true);
        filter.setOutputWordCounts(true);
        filter.setTFTransform(true);
        filter.setIDFTransform(true);
        filter.setStopwords(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\StopWords.txt"));
        filter.setTokenizer(tokenizer);
        filter.setStemmer(stemmer);
        System.out.println("%%%%%%%%%%%%%%"+filter.getStemmer());
        
        //import data from file
        TextDirectoryLoader docLoader = new TextDirectoryLoader();
        docLoader.setDirectory(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\News"));
        Instances data = docLoader.getDataSet();
        System.out.println("\n\nImported data:\n\n" + data);

        // apply the StringToWordVector filter
        filter.setInputFormat(data);
        Instances dataFiltered = Filter.useFilter(data, filter);
        System.out.println("###########"+dataFiltered.numAttributes());
        System.out.println("\n\nFiltered data:\n\n" + dataFiltered);
     
        //initialize the model and set SVM type and kernal type
        LibSVM svm = new LibSVM();
        //-S 1 -K 3 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.0010 -P 0.1
        String svmOptions = "-S 1 -K 3";
        svm.setOptions(weka.core.Utils.splitOptions(svmOptions));
        
//        Tag t1=new Tag(1,"SVMTYPE_NU_SVC");
//        Tag t2=new Tag(3,"SVMTYPE_EPSILON_SVR");
//        Tag[] ts={t1,t2};
//        SelectedTag st=new SelectedTag(3, ts);
//        System.out.println(st.getSelectedTag().getID());
//        svm.setSVMType(st);
        System.out.println("&&&&&&&&"+svm.getSVMType()+svm.getKernelType());//1,3 best result 81%
       
        //perform cross vlaidation
        Evaluation evaluation = new Evaluation(dataFiltered);
        evaluation.crossValidateModel(svm, dataFiltered, 2, new Random(1));
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.weightedAreaUnderROC());
    }
    
    public void testFilter() throws Exception{
        StringToWordVector filter = new StringToWordVector();
        String[] options = {"-C","-T","-I","-S","-L"};
        filter.setOptions(options);
        System.out.println(filter.getLowerCaseTokens());
    }
}
