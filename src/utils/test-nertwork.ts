/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { NeuralNetwork } from "../core";
import { TestingNetworkData } from "../models";

export function testNetwork(
  network: NeuralNetwork,
  testingNetworkData: TestingNetworkData[]
): void {
  console.log(`\nTesting of ${network.trainingName}:`);
  testingNetworkData.forEach((data) => {
    const output: number[] = network.predict(data.inputs);
    console.log("Inputs:",data.inputs,"Output:",output.map((o) => parseFloat(o.toFixed(2)))
    );
  });
}
