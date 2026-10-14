/*
 * Experiment 9 – PageRank Algorithm Implementation in Java
 * Student: Tanaya Salunke | Roll: 23101A0074 | INFT-A
 */
import java.io.*;
import java.util.*;

public class Exp9PageRank {
    private static final int NUM_NODES = 1000; // number of vertices
    private static final int NUM_EDGES = 5000; // total directed edges
    private static final double DAMPING_FACTOR = 0.85;
    private static final int MAX_ITER = 20;
    private static final double TOL = 1e-6;

    public static void main(String[] args) throws IOException {
        // 1. Generate synthetic directed graph (adjacency list)
        List<Set<Integer>> outLinks = new ArrayList<>(NUM_NODES);
        List<Set<Integer>> inLinks = new ArrayList<>(NUM_NODES);
        for (int i = 0; i < NUM_NODES; i++) {
            outLinks.add(new HashSet<>());
            inLinks.add(new HashSet<>());
        }
        Random rnd = new Random(42);
        int edgesAdded = 0;
        while (edgesAdded < NUM_EDGES) {
            int src = rnd.nextInt(NUM_NODES);
            int dst = rnd.nextInt(NUM_NODES);
            if (src == dst) continue; // avoid self-loop
            if (outLinks.get(src).add(dst)) { // only add if new edge
                inLinks.get(dst).add(src);
                edgesAdded++;
            }
        }
        System.out.println("Generated graph with " + NUM_NODES + " nodes and " + edgesAdded + " edges.");

        // 2. Initialise PageRank values
        double[] pr = new double[NUM_NODES];
        double initRank = 1.0 / NUM_NODES;
        Arrays.fill(pr, initRank);

        // 3. Iterative computation
        double[] newPr = new double[NUM_NODES];
        double teleport = (1.0 - DAMPING_FACTOR) / NUM_NODES;
        for (int iter = 1; iter <= MAX_ITER; iter++) {
            // compute contributions
            Arrays.fill(newPr, teleport);
            for (int v = 0; v < NUM_NODES; v++) {
                Set<Integer> out = outLinks.get(v);
                if (out.isEmpty()) {
                    // dangling node distributes its rank to all nodes
                    double share = DAMPING_FACTOR * pr[v] / NUM_NODES;
                    for (int i = 0; i < NUM_NODES; i++) {
                        newPr[i] += share;
                    }
                } else {
                    double share = DAMPING_FACTOR * pr[v] / out.size();
                    for (int w : out) {
                        newPr[w] += share;
                    }
                }
            }
            // check convergence
            double diff = 0.0;
            for (int i = 0; i < NUM_NODES; i++) {
                diff += Math.abs(newPr[i] - pr[i]);
            }
            System.out.printf("Iteration %2d: total change = %.6f%n", iter, diff);
            // copy back
            System.arraycopy(newPr, 0, pr, 0, NUM_NODES);
            if (diff < TOL) {
                System.out.println("Converged after " + iter + " iterations.");
                break;
            }
        }

        // 4. Extract top 10 nodes
        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < NUM_NODES; i++) idx.add(i);
        idx.sort((a, b) -> Double.compare(pr[b], pr[a])); // descending
        System.out.println("Top 10 PageRank nodes:");
        for (int i = 0; i < 10; i++) {
            int node = idx.get(i);
            System.out.printf("  Node %4d : %.6f%n", node, pr[node]);
        }

        // 5. Save results to a text file for the doc generator
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("exp9_results.txt")))) {
            pw.println("NUM_NODES=" + NUM_NODES);
            pw.println("NUM_EDGES=" + edgesAdded);
            pw.println("DAMPING_FACTOR=" + DAMPING_FACTOR);
            pw.println("MAX_ITER=" + MAX_ITER);
            pw.println("TOP_10_RANKS");
            for (int i = 0; i < 10; i++) {
                int node = idx.get(i);
                pw.printf("%d %.6f%n", node, pr[node]);
            }
        }
        System.out.println("Results saved to exp9_results.txt");
    }
}
