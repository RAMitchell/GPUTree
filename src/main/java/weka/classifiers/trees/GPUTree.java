package weka.classifiers.trees;

import com.sun.jna.Memory;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.classifiers.AbstractClassifier;

import com.sun.jna.*;

import java.util.Enumeration;
import java.util.Vector;
import java.util.List;
import java.util.Arrays;

/**
 * Created by R on 17/04/2016.
 */
public class GPUTree extends AbstractClassifier {

    /**
     * Native library definitions
     */
    public interface CLibrary extends Library {
    CLibrary clib = (CLibrary) Native.loadLibrary("lib/GPUTree", CLibrary.class);

        public static class TreeNode extends Structure {

            public static class ByReference extends TreeNode implements Structure.ByReference {
            }

            public int attributeIndex;
            public float attributeValue;
            public float infogain;
            public float leftProb;
            public float rightProb;

            @Override
            protected List getFieldOrder() {
                return Arrays.asList("attributeIndex", "attributeValue", "infogain", "leftProb", "rightProb");
            }
        }

        boolean test_cuda();

        void  force_context_init();

        int generate_tree(Pointer in_attributes, int n_attributes, Pointer in_classes, int n_instances, int n_levels, TreeNode.ByReference out_tree);

    }

    /** Upper bound on the tree depth */
    protected int m_MaxDepth = 3;

    /** The decision tree */
    CLibrary.TreeNode[] m_Tree;

    /** Attribute names. Used by toString() */
    String[] m_AttributeNames;

    /**
     * Preload the cuda run-time
     */
    @Override
    public void preExecution(){
        if(CLibrary.clib.test_cuda()) {
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
        result.enable(Capability.BINARY_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }


    public void buildClassifier(Instances instances) throws Exception {

        //Is there a cuda capable GPU?
        if (!CLibrary.clib.test_cuda()) {
            throw new WekaException("CUDA capable GPU required for this classifier.");
        }

        // can classifier handle the data?
        getCapabilities().testWithFail(instances);

        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        //Store attribute names
        m_AttributeNames = new String[instances.numAttributes() - 1];
        int counter = 0;
        for(int i = 0; i < instances.numAttributes() - 1; i++){
            if(i != instances.classIndex()){
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

        //Call native classifier
        CLibrary.clib.generate_tree(attributes, instances.numAttributes() - 1, classes, instances.numInstances(), m_MaxDepth, treeRef);

    }

    private String toStringRecurse(int index, int depth){

        if( depth >= m_MaxDepth){
            return "";
        }

        StringBuffer text = new StringBuffer();

        CLibrary.TreeNode node = m_Tree[index];

        if( node.attributeIndex  == -1){
            return "";
        }

        for(int i = 0; i < depth; i++) {
            text.append("|   ");
        }
        text.append(m_AttributeNames[node.attributeIndex]  + " < "  + Utils.doubleToString(node.attributeValue,2) + "\n");

        text.append(toStringRecurse(index * 2 + 1, depth + 1));

        for(int i = 0; i < depth; i++) {
            text.append("|   ");
        }
        text.append(m_AttributeNames[node.attributeIndex]  + " >= "  + Utils.doubleToString(node.attributeValue,2) + "\n");
        text.append(toStringRecurse(index * 2 + 2, depth + 1));

        return text.toString();
    }

    @Override
    public String toString() {
        return toStringRecurse(0,0);
    }

    /**
     * @return  Maximum number of nodes our binary tree can contain
     */
     private int maxNodes(){
         return (1<<m_MaxDepth) - 1;
     }

    /**
     * Recursively traverses the tree for a given instance and returns the class probability.
     * @param instance
     * @param nodeIndex
     * @return Class probability
     */
    private float distributionRecurse(Instance instance, int nodeIndex){

        float split = m_Tree[nodeIndex].attributeValue;

        //Account for the fact that our attribute index does not include the class
        int nativeIndex = m_Tree[nodeIndex].attributeIndex;
        int attributeIndex = nativeIndex< instance.classIndex()?nativeIndex:nativeIndex + 1;

        if( instance.value(attributeIndex) < split){
            int nextIndex = nodeIndex * 2 + 1;

            if( nextIndex >= maxNodes()){
                return m_Tree[nodeIndex].leftProb;
            }
            else if(m_Tree[nextIndex].attributeIndex ==  - 1){
                return m_Tree[nodeIndex].leftProb;
            }
            else{
                return distributionRecurse(instance, nextIndex);
            }


        }
        else{

            int nextIndex = nodeIndex * 2 + 2;

            if( nextIndex >= maxNodes()){
                return m_Tree[nodeIndex].rightProb;
            }
            else if(m_Tree[nextIndex].attributeIndex ==  - 1){
                return m_Tree[nodeIndex].rightProb;
            }
            else{
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

        double[] probs = new double[2];

        probs[1] = distributionRecurse(instance,0);
        probs[0] = 1 - probs[1];

        return probs;
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
