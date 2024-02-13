import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Main {

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Documentación:");
            System.out.println("java -jar estimacionNaiveBayes5fCV.jar /path/data.arff /path/results.txt");
        }

        try {
            // Argumentos
            String rutaArchivoARFF = args[0];
            String rutaArchivoResultados = args[1];

            // Cargar datos desde el archivo ARFF
            Instances datos = new Instances(new java.io.FileReader(rutaArchivoARFF));
            datos.setClassIndex(datos.numAttributes() - 1);

            // Entrenar modelo Naive Bayes y kfcv5
            Evaluation evaluacion = entrenarNaiveBayesConValidacionCruzada(datos, 5);

            // Obtener resultados de la evaluación
            String resultados = evaluacion.toMatrixString("Matriz de Confusión") + "\n";
            resultados += "Métricas de Precisión:\n" + evaluacion.toClassDetailsString();

            // Crear archivo de resultados
            guardarResultadosEnArchivo(rutaArchivoResultados, rutaArchivoARFF, resultados);

            System.out.println("Ejecución exitosa. Resultados guardados en: " + rutaArchivoResultados);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Evaluation entrenarNaiveBayesConValidacionCruzada(Instances datos, int numFolds) throws Exception {
        //Naive Bayes
        NaiveBayes naiveBayes = new NaiveBayes();

        // kfcv5
        Evaluation evaluacion = new Evaluation(datos);
        for(int i=1;i<=numFolds;i++){
            evaluacion.crossValidateModel(naiveBayes, datos, numFolds, new java.util.Random(i));
        }
        return evaluacion;
    }

    private static void guardarResultadosEnArchivo(String rutaArchivoResultados, String rutaArchivoARFF, String resultados) throws Exception {
        // Crear el archivo de resultados
        BufferedWriter writer = new BufferedWriter(new FileWriter(rutaArchivoResultados));
        // Escribir la fecha de ejecución y los argumentos
        writer.write("Fecha de ejecución: " + obtenerFechaActual() + "\n");
        writer.write("Argumentos de ejecución:\n");
        writer.write("Archivo ARFF: " + rutaArchivoARFF + "\n");
        writer.write("Archivo de Resultados: " + rutaArchivoResultados + "\n\n");
        // Escribir los resultados de la evaluación
       writer.write(resultados);
        writer.close();
    }

    private static String obtenerFechaActual() {
        SimpleDateFormat formatter = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
        Date date = new Date();
        return formatter.format(date);
    }
}

