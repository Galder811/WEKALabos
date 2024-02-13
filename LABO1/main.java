import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class Main {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        System.out.println("1-Aztertzen ari garen fitxategiko path-a: "+ args[0]);
        System.out.println("2-Instantzia kopurua: "+ data.numInstances());
        System.out.println("3-Atributu kopurua: "+data.numAttributes());
        System.out.println("4-Lehenengo atributuak har ditzakeen balio ezberdinak (balio ezberdin kopurua): "+data.numDistinctValues(0));
        System.out.println("5-Azken atributuak hartzen dituen balioak eta beraien maiztasuna:");
        int ind=0;
        for(int i : data.attributeStats(data.numAttributes()-2).nominalCounts){
            System.out.println("Izena: "+data.attribute(data.numAttributes()-2).value(ind++)+" || Maiztasuna: "+i);
        }
        System.out.println("");
        System.out.println("6-Zein da klase minoritarioa: ");
        String minKlas="";
        int minKlasInt=data.numInstances()+1;
        for (int i =0;i<data.attributeStats(data.numAttributes()-1).nominalCounts.length;i++){
            if (minKlasInt>data.attributeStats(data.numAttributes() - 1).nominalCounts[i]) {
                minKlasInt = data.attributeStats(data.numAttributes() - 1).nominalCounts[i];
                minKlas = data.attribute(data.numAttributes() - 1).value(i);
            }
        }
        System.out.println("Klase minoritarioa: " +minKlas+" || Maiztasuna: "+minKlasInt);
        System.out.println("");
        System.out.println("7-Azken aurreko atributuak dituen missing value kopurua: "+data.attributeStats(data.numAttributes()-3).missingCount);
    }
}
