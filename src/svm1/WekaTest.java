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
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.TextDirectoryLoader;
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
        System.out.println(v[0]+" "+v[1]);

    }

    public void loadFromTextFiles() throws IOException, Exception {
        
        StringToWordVector filter = new StringToWordVector();
        String[] options = {"-C"};
        filter.setOptions(options);
        
        // convert the directory into a dataset
        TextDirectoryLoader trainingDocLoader = new TextDirectoryLoader();
        trainingDocLoader.setDirectory(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\text"));
        Instances trainingData = trainingDocLoader.getDataSet();
        System.out.println("\n\nImported data:\n\n" + trainingData);

        // apply the StringToWordVector
        filter.setInputFormat(trainingData);
        Instances trainingDataFiltered = Filter.useFilter(trainingData, filter);
        System.out.println("\n\nFiltered data:\n\n" + trainingDataFiltered);
        System.out.println(trainingDataFiltered.classIndex());
        
        LibSVM svm = new LibSVM();
        svm.buildClassifier(trainingDataFiltered);
        
        TextDirectoryLoader testDocSetLoader = new TextDirectoryLoader();
        testDocSetLoader.setDirectory(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\textTest"));
        Instances testData = testDocSetLoader.getDataSet();
        System.out.println("\n\nImported data:\n\n" + testData);
        
        
        filter.setInputFormat(testData);
        Instances testDataFiltered = Filter.useFilter(testData, filter);
        System.out.println("\n\nFiltered data:\n\n" + testDataFiltered);
        System.out.println(testDataFiltered.instance(1));
        
       double[] predictions=svm.distributionForInstance(testDataFiltered.instance(1));
      //  System.out.println(predictions[0]+"  "+predictions[1]);
        
    }
    
    
}
