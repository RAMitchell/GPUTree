package weka.classifiers.trees;

import com.sun.jna.Memory;
import weka.classifiers.rules.ZeroR;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.classifiers.AbstractClassifier;

import com.sun.jna.*;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Vector;
import java.util.List;
import java.util.Arrays;

/**
 * Created by Rory Mitchell on 17/04/2016.
 */
public class GPUTree extends AbstractClassifier {

    private static final long serialVersionUID = -1775960263491961836L;

    /**
     * Native library definitions
     */
    public interface CLibrary extends Library {

        CLibrary clib = (CLibrary) Native.loadLibrary(WekaPackageManager.PACKAGES_DIR.toString() + "/GPUTree/lib/GPUTree", CLibrary.class);

        class TreeNode extends Structure implements Serializable {

            private static final long serialVersionUID = 6889219211806337654L;

            public static class ByReference extends TreeNode implements Structure.ByReference {
            }

            public int attributeIndex;
            public float attributeValue;
            public Pointer distribution;
            public Pointer  counts;

            /**
             * Allocate memory for native code to store distribution
             *
             * @param numValues Number of values class attribute can take
             */
            public void allocateDistribution(int numValues) {
                distribution = new Memory(numValues * Native.getNativeSize(Float.TYPE));
            }
            /**
             * Allocate memory for native code to store counts
             *
             * @param numValues Number of values class attribute can take
             */
            public void allocateCounts(int numValues) {
                 counts = new Memory(numValues * Native.getNativeSize(Integer.TYPE));
            }

            public double[] getDistribution(int numValues) {
                double[] dist = new double[numValues];
                for (int i = 0; i < numValues; i++) {
                    dist[i] = distribution.getFloat(i * Native.getNativeSize(Float.TYPE));
                }

                return dist;
            }

            public int sumCounts(int numValues){
                int sum = 0;
                for (int i = 0; i < numValues; i++) {
                    sum  += counts.getInt(i * Native.getNativeSize(Integer.TYPE));
                }
                return sum;
            }

            @Override
            protected List getFieldOrder() {
                return Arrays.asList("attributeIndex", "attributeValue","distribution","counts");
            }
        }

        boolean test_cuda();

        void force_context_init();

        int generate_tree(Pointer in_attributes, int n_attributes, Pointer in_classes, int n_instances, int n_class_values, int n_levels, TreeNode.ByReference out_tree);

    }

    /**
     * Upper bound on the tree depth
     */
    protected int m_MaxDepth = 3;

    /**
     * Number of nominal class values
     */
    protected int m_NumClasses = 0;
    /**
     * The decision tree
     */
    CLibrary.TreeNode[] m_Tree;

    /**
     * Attribute names. Used by toString()
     */
    protected String[] m_AttributeNames;

    /**
     * ZeroR model that is used if no attributes are present.
     */
    protected ZeroR m_zeroR;

    /**
     * Preload the cuda run-time
     */
    @Override
    public void preExecution() {
        if (CLibrary.clib.test_cuda()) {
            CLibrary.clib.force_context_init();
        }
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();


        // attributes
        result.enable(Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }


    public void buildClassifier(Instances instances) throws Exception {

        m_NumClasses = instances.numClasses();

        //Is there a cuda capable GPU?
        if (!CLibrary.clib.test_cuda()) {
            throw new WekaException("CUDA capable GPU required for this classifier.");
        }

        // can classifier handle the data?
        getCapabilities().testWithFail(instances);

        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        if (instances.size() == 0) {
            return;
        }

        m_zeroR = null;
        if (instances.numAttributes() == 1) {
            m_zeroR = new ZeroR();
            m_zeroR.buildClassifier(instances);
            return;
        }

        //Store attribute names
        m_AttributeNames = new String[instances.numAttributes() - 1];
        int counter = 0;
        for (int i = 0; i < instances.numAttributes() - 1; i++) {
            if (i != instances.classIndex()) {
                m_AttributeNames[counter] = instances.attribute(i).name();
                counter++;
            }
        }

        //Copy attributes and classes into plain arrays to pass on to C function
        Pointer attributes = new Memory((instances.numAttributes() - 1) * instances.numInstances() * Native.getNativeSize(Float.TYPE));
        Pointer classes = new Memory(instances.numInstances() * Native.getNativeSize(Byte.TYPE));

        int attributes_position = 0;

        for (int i = 0; i < instances.numAttributes(); i++) {
            double[] a = instances.attributeToDoubleArray(i);

            if (i == instances.classIndex()) {
                for (int j = 0; j < instances.numInstances(); j++) {
                    classes.setByte(j * Native.getNativeSize(Byte.TYPE), (byte) a[j]);
                }
            } else {
                for (int j = 0; j < instances.numInstances(); j++) {
                    attributes.setFloat(attributes_position++ * Native.getNativeSize(Float.TYPE), (float) a[j]);
                }
            }
        }


        //Allocate memory for tree
        CLibrary.TreeNode.ByReference treeRef = new CLibrary.TreeNode.ByReference();
        m_Tree = (CLibrary.TreeNode[]) treeRef.toArray(maxNodes());
        for (int i = 0; i < maxNodes(); i++) {
            m_Tree[i].allocateDistribution(instances.numClasses());
            m_Tree[i].allocateCounts(instances.numClasses());
        }

        //Call native classifier
        CLibrary.clib.generate_tree(attributes, instances.numAttributes() - 1, classes, instances.numInstances(), instances.numClasses(), m_MaxDepth, treeRef);

    }

    private String toStringLeaf(CLibrary.TreeNode node){

        //Find most probable class
        double max = 0;
        int max_index = 0;
        double dist[] = node.getDistribution(m_NumClasses);
        for (int i = 0; i < m_NumClasses; i++) {
            if (dist[i] > max) {
                max = dist[i];
                max_index = i;
            }
        }
        return " : c" + max_index + " ("+ node.sumCounts(m_NumClasses) + "/" + (node.sumCounts(m_NumClasses) - node.counts.getInt(max_index * Native.getNativeSize(Integer.TYPE)))+")\n";
    }

    private String toStringRecurse(int index, int depth) {

        if (depth > m_MaxDepth) {
            return "";
        }

        StringBuffer text = new StringBuffer();

        CLibrary.TreeNode node = m_Tree[index];

        if (node.attributeIndex == -1) {
            return toStringLeaf(node);
        }

        text.append("\n");

        for (int i = 0; i < depth; i++) {
            text.append("|   ");
        }
        text.append(m_AttributeNames[node.attributeIndex] + " < " + Utils.doubleToString(node.attributeValue, 2));

        text.append(toStringRecurse(index * 2 + 1, depth + 1));

        for (int i = 0; i < depth; i++) {
            text.append("|   ");
        }
        text.append(m_AttributeNames[node.attributeIndex] + " >= " + Utils.doubleToString(node.attributeValue, 2));
        text.append(toStringRecurse(index * 2 + 2, depth + 1));

        return text.toString();
    }

    @Override
    public String toString() {
        if (m_Tree == null) {
            return "";
        }
        return toStringRecurse(0, 0);
    }

    /**
     * @return Maximum number of nodes our binary tree can contain
     */
    private int maxNodes() {
        return (1 << (m_MaxDepth + 1)) - 1;
    }

    /**
     * Recursively traverses the tree for a given instance and returns the class probability.
     *
     * @param instance
     * @param nodeIndex
     * @return Class probability
     */
    private double[] distributionRecurse(Instance instance, int nodeIndex) {

        int nativeIndex = m_Tree[nodeIndex].attributeIndex;
        if (nativeIndex == -1) {
            return m_Tree[nodeIndex].getDistribution(instance.numClasses());
        }

        float split = m_Tree[nodeIndex].attributeValue;

        //Account for the fact that our attribute index does not include the class
        int attributeIndex = nativeIndex < instance.classIndex() ? nativeIndex : nativeIndex + 1;

        if (instance.value(attributeIndex) < split) {
            int nextIndex = nodeIndex * 2 + 1;

            if (nextIndex >= maxNodes()) {
                return m_Tree[nodeIndex].getDistribution(instance.numClasses());
            } else {
                return distributionRecurse(instance, nextIndex);
            }

        } else {

            int nextIndex = nodeIndex * 2 + 2;

            if (nextIndex >= maxNodes()) {
                return m_Tree[nodeIndex].getDistribution(instance.numClasses());
            } else {
                return distributionRecurse(instance, nextIndex);
            }
        }
    }

    /**
     * Calculates the class membership probabilities for the given test
     * instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @throws Exception if there is a problem generating the prediction
     */
    public double[] distributionForInstance(Instance instance)
            throws Exception {

        if (m_zeroR != null) {
            return m_zeroR.distributionForInstance(instance);
        }

        if (m_Tree == null) {
            return new double[instance.numClasses()];
        }
        return distributionRecurse(instance, 0);
    }

    public String globalInfo() {
        return "Fast GPU decision tree classifier.";
    }

    /**
     * Parses a given list of options. <p/>
     * <p>
     * <!-- options-start -->
     * Valid options are: <p/>
     * <p>
     * <pre> -L
     *  Maximum tree depth (default 3)</pre>
     * <p>
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */

    public void setOptions(String[] options) throws Exception {

        String depthString = Utils.getOption('L', options);
        if (depthString.length() != 0) {
            m_MaxDepth = Integer.parseInt(depthString);
        } else {
            m_MaxDepth = 3;
        }
        Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current settings of the classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {

        String[] options = new String[2];
        int current = 0;
        options[current++] = "-L";
        options[current++] = "" + getMaxDepth();

        return options;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(1);

        newVector.
                addElement(new Option("\tMaximum tree depth (default 3)",
                        "L", 1, "-L"));

        return newVector.elements();
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String maxDepthTipText() {
        return "The maximum tree depth";
    }

    /**
     * Get the value of MaxDepth.
     *
     * @return Value of MaxDepth.
     */
    public int getMaxDepth() {

        return m_MaxDepth;
    }

    /**
     * Set the value of MaxDepth.
     *
     * @param newMaxDepth Value to assign to MaxDepth.
     */
    public void setMaxDepth(int newMaxDepth) {

        m_MaxDepth = newMaxDepth;
    }

    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    public static void main(String[] argv) {
        runClassifier(new GPUTree(), argv);
    }
}
