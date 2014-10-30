/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package svm1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
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
//        BufferedReader reader = new BufferedReader(new FileReader("C:\\Users\\hp\\Desktop\\SVM implementation\\iris.arff"));
//        Instances data = new Instances(reader);
        DataSource source = new DataSource("C:\\Users\\hp\\Desktop\\SVM implementation\\iris.arff");
        System.out.println(source.getStructure());
        Instances data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        System.out.println(data);
        //initialize svm classifier
        LibSVM svm = new LibSVM();
        svm.buildClassifier(data);

    }

    public void loadFromTextFiles() throws IOException, Exception {
        // convert the directory into a dataset
        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory(new File("C:\\Users\\hp\\Desktop\\SVM implementation\\text"));
        Instances dataRaw = loader.getDataSet();
        System.out.println("\n\nImported data:\n\n" + dataRaw);

        // apply the StringToWordVector
        
        StringToWordVector filter = new StringToWordVector();
        String[] options={"-C"};
        filter.setOptions(options);
        filter.setInputFormat(dataRaw);
        Instances dataFiltered = Filter.useFilter(dataRaw, filter);
        System.out.println("\n\nFiltered data:\n\n" + dataFiltered);
    }
}
