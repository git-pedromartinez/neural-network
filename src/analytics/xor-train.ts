import profiler from "v8-profiler-next";
import fs from "fs";
import { NeuralNetwork } from "../core";
import { totalTime } from "../utils";
import { network_config, XOR_DATA, XOR_TRAINING } from "./network-config";

// Start CPU profiling
profiler.startProfiling("CPU Profile", true);

// Your code here (e.g., training the neural network)
const startTime: Date = new Date();
const network = new NeuralNetwork(network_config);
network.trainingName = XOR_TRAINING;
network.errorThreshold = 0.001;
network.showLogs = true;
console.log("Training started at:", startTime.toLocaleString());
network.train(XOR_DATA);
console.log("Total time:", totalTime(startTime), "seconds");
network.saveTraining();

// Stop CPU profiling
const profile = profiler.stopProfiling("CPU Profile");

// Save the profile to a file
profile.export((error, result) => {
  if (result) {
    fs.writeFileSync("cpu-profile.cpuprofile", result);
  }
  profile.delete();
});

// You can now open the file with https://www.speedscope.app/
