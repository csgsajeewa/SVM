/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package svm1;

import libsvm.svm_model;

/**
 *
 * @author hp
 */
public class SVM1 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
//        SVMTest svmt=new SVMTest();
//        svm_model model=svmt.svmTrain();
//        double[] features={1,0,550};
//        double v=svmt.predict(features, model);
//        System.out.println(v);
//        svmt.evaluate(features, model);
        WekaTest wt=new WekaTest();
        wt.testCrossValidataion();
        
    }
}
