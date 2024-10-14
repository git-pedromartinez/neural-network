/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { NeuralNetwork } from "../../core";
import { totalTime } from "../../utils";
import { network_config } from "./network-config";
import { XOR_DATA } from "./data";
import { XOR_TRAINING } from "./constants";

const network: NeuralNetwork = new NeuralNetwork(network_config);
const inputs = [1, 0];
let startTime: Date;

network.trainingName = XOR_TRAINING;
console.log(network.trainingName);

// startTime = new Date();
// console.log("Training started at:", startTime.toLocaleString());
// network.train(XOR_DATA);
// console.log("Total time:", totalTime(startTime), "seconds");
// network.saveTraining();

startTime = new Date();
network.loadTraining();
console.log("Prediction started at:", startTime.toLocaleString());
console.log("Input:", inputs, "Result:", network.predict(inputs));
console.log("Total time:", totalTime(startTime), "seconds");
