import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;
import java.io.BufferedWriter;
import java.io.FileWriter;

public class Main {

    public static void main(String[] args) throws Exception {
        // Verificar los argumentos de entrada
        if (args.length != 3) {
            System.err.println("Uso: java NaiveBayesPrediction <NB.model> <test_blind.arff> <test_predictions.txt>");
            System.exit(1);
        }

        // Cargar el modelo Naive Bayes entrenado
        Classifier nb = (Classifier) SerializationHelper.read(args[0]);

        // Cargar las instancias de prueba
        DataSource source = new DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        // Crear archivo de salida para las predicciones
        String rutaArchivoResultados = args[2];
        BufferedWriter writer = new BufferedWriter(new FileWriter(rutaArchivoResultados));

        // Hacer predicciones para cada instancia en el conjunto de prueba
        writer.write("Instancia, Clase Predicha\n");
        for (int i = 0; i < data.numInstances(); i++) {
            double pred = nb.classifyInstance(data.instance(i));
            writer.write((i+1) + ", " + data.classAttribute().value((int) pred) + "\n");
        }

        writer.close();
    }
}
