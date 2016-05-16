
package weka.classifiers.trees;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;

import junit.framework.Test;
import junit.framework.TestSuite;
import weka.core.Instance;
import weka.core.Instances;
import weka.datagenerators.DataGenerator;
import weka.datagenerators.classifiers.classification.RDG1;

/**
 * Tests GPUTree. Run from the command line with:<p>
 * java weka.classifiers.trees.GPUTreeTest
 */
public class GPUTreeTest extends AbstractClassifierTest {

    public GPUTreeTest(String name) {
        super(name);
    }

    public Classifier getClassifier() {
        return new GPUTree();
    }


    public static void benchmark() throws Exception {

        int nExamples = 1000;

        //Number of timing runs to average
        final int testIterations = 10;
        //Number runs to warm up JIT compiler
        final int hotspotIterations = 2;

        final int nAttributes = 10;
        final int maxLevels = 4;

        System.out.println("Levels, Instances, GPUTree, REPTree, speedup");

        for (int i = 0; i < 8; i++) {

            //Generate random instances
            RDG1 generator = new RDG1();
            generator.setMaxRuleSize(50);
            generator.setNumAttributes(nAttributes);
            generator.setNumNumeric(nAttributes);
            generator.setNumExamples(nExamples);
            generator.defineDataFormat();
            Instances instances = generator.generateExamples();


            GPUTree gpuTree = new GPUTree();
            long gpuElapsed = 0;
            for (int j = 0; j < testIterations + hotspotIterations; j++) {

                gpuTree = new GPUTree();
                gpuTree.setMaxDepth(maxLevels);
                gpuTree.preExecution();
                long start = System.currentTimeMillis();
                gpuTree.buildClassifier(instances);
                if (j > hotspotIterations) {
                    gpuElapsed += System.currentTimeMillis() - start;
                }
            }
            double gpuSeconds = (double) gpuElapsed / 1000 / testIterations;
            System.out.print(maxLevels + ", ");
            System.out.print(nExamples + ", ");
            System.out.print(String.format("%1.2f", gpuSeconds) + ", ");


            REPTree repTree = new REPTree();
            long repElapsed = 0;
            for (int j = 0; j < testIterations + hotspotIterations; j++) {
                repTree = new REPTree();
                repTree.setMaxDepth(maxLevels);
                repTree.setNoPruning(true);
                repTree.setMinNum(0);
                long start = System.currentTimeMillis();
                repTree.buildClassifier(instances);
                if (j > hotspotIterations) {
                    repElapsed += System.currentTimeMillis() - start;
                }
            }

            double repSeconds = (double) repElapsed / 1000 / testIterations;
            System.out.print(String.format("%1.2f", repSeconds) + ", ");

            System.out.print(String.format("%1.2f", repSeconds / gpuSeconds) + "\n");

            //Check correctness
            for (Instance inst : instances) {
                if (gpuTree.classifyInstance(inst) != repTree.classifyInstance(inst)) {
                    System.out.println("GPUTree prediction: " + gpuTree.classifyInstance(inst));
                    System.out.println("REPTree prediction: " + repTree.classifyInstance(inst));
                    System.out.println("GPUTree tree:");
                    System.out.println(gpuTree.toString());
                    System.out.println("REPTree tree:");
                    System.out.println(repTree.toString());
                    System.exit(-1);
                }
            }

            nExamples *= 2;
        }
    }

    public static Test suite() {
        return new TestSuite(GPUTreeTest.class);
    }


    public static void main(String[] args) throws Exception {
        junit.textui.TestRunner.run(suite());
        benchmark();
    }

}
