/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { Layer } from "./layer";
import { ActivationFunctions } from "./activation-functions";
import {
  ActivationFunction,
  ActivationFunctionDerivative,
  Matrix,
  NetworkMetadata,
  NetworkSimulatorConfig,
  NeuralNetworkConfig,
  NeuralNetworkData,
  TrainingNetworkData,
} from "../models";
import { StorageManager } from "../utils";

const NetworkStore = new StorageManager<TrainingNetworkData>();
const NetworkHistoryStore = new StorageManager<any[]>();

export class NeuralNetwork {
  private history: any[] = [];

  private layers!: Layer[];
  private learningRate!: number;
  private epochs!: number;

  /**
   * @variable `_trainingName` defines the name of the training, this name is used to save and load the training in the training history
   */
  private _trainingName: string = `NeuralNetworkTraining_${new Date().getTime()}`;
  public get trainingName(): string {
    return this._trainingName;
  }
  public set trainingName(value: string) {
    this._trainingName = value;
  }

  /**
   * @variable `_errorThreshold` defines the error threshold, this variable can be adjusted to improve the model's accuracy, default is 0.01
   */
  private _errorThreshold: number = 0.01;
  public get errorThreshold(): number {
    return this._errorThreshold;
  }
  public set errorThreshold(value: number) {
    this._errorThreshold = value;
  }

  /**
   * @variable `_showLogs` defines whether logs are shown in the console, default is false, and can be adjusted at runtime
   * @default false
   */
  private _showLogs: boolean = false;
  public get showLogs(): boolean {
    return this._showLogs;
  }
  public set showLogs(value: boolean) {
    this._showLogs = value;
  }

  /**
   * Experimental feature: simulator
   */
  private metaData: NetworkMetadata = {};

  public saveHistory: boolean = false;

  constructor({
    sizes,
    learningRate,
    epochs,
    activationFunction = ActivationFunctions.sigmoid,
    activationDerivative = ActivationFunctions.sigmoidDerivative,
  }: NeuralNetworkConfig) {
    const config: NeuralNetworkConfig = {
      sizes,
      learningRate,
      epochs,
      activationFunction,
      activationDerivative,
    };

    this.updateMetaData(config);
    this.loadNetworkConfig(config);
  }

  /**
   * Experimental feature: simulator
   */
  private loadNetworkConfig(
    config?: NeuralNetworkConfig | NetworkMetadata
  ): void {
    if (
      config?.sizes?.length &&
      config?.sizes?.length > 1 &&
      config?.learningRate &&
      config?.epochs
    ) {
      this.layers = [];
      for (let i = 1; i < config.sizes.length; i++) {
        this.layers.push(
          new Layer(
            config.sizes[i],
            config.sizes[i - 1],
            config.activationFunction!,
            config.activationDerivative!,
            i
          )
        );
      }
      this.learningRate = config.learningRate;
      this.epochs = config.epochs;
      if ((config as NetworkMetadata)?.trainingName !== undefined) {
        this.trainingName = (config as NetworkMetadata)?.trainingName!;
      }
      if ((config as NetworkMetadata)?.errorThreshold !== undefined) {
        this.errorThreshold = (config as NetworkMetadata).errorThreshold!;
      }
    } else {
      this.log("Invalid network configuration:", config);
      if (!(config?.sizes?.length && config.sizes.length > 1)) {
        this.log(
          "Please provide a sizes valid setting. current:",
          config?.sizes
        );
      }
      if (!config?.learningRate) {
        this.log(
          "Please provide a learningRate valid setting. current:",
          config?.learningRate
        );
      }
      if (!config?.epochs) {
        this.log(
          "Please provide a epochs valid setting. current:",
          config?.epochs
        );
      }
    }
  }

  /**
   * Experimental feature: simulator
   */
  private updateMetaData(
    metaData?: NeuralNetworkConfig | NetworkMetadata
  ): void {
    this.metaData = {
      ...this.metaData,
      ...metaData,
      trainingName: this.trainingName,
      errorThreshold: this.errorThreshold,
    };
    this.log("Network metadata updated:", this.metaData);
  }

  train(TrainingNetworkData: NeuralNetworkData[]): void {
    const inputs: Matrix<"2D"> = TrainingNetworkData.map((d) => d.inputs);
    const targets: Matrix<"2D"> = TrainingNetworkData.map((d) => d.targets);

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      inputs.forEach((input, index) => {
        this.backward(epoch, input, targets[index]);
      });
    }
    this.log(`Training completed after ${this.epochs} epochs.`);

    if (this.saveHistory) {
      NetworkHistoryStore.showLogs = this.showLogs;
      NetworkHistoryStore.saveObject(
        this.history,
        this.trainingName + "_history"
      );
    }
  }

  private backward(epoch: number, inputs: Matrix, targets: Matrix): void {
    const outputs: Matrix<"2D"> = this.forward(inputs);
    const outputErrors: Matrix = targets.map(
      (t, i) => t - outputs[outputs.length - 1][i]
    );
    const errors: Matrix<"2D"> = this.calculateErrors(outputErrors);

    for (let i = this.layers.length - 1; i >= 0; i--) {
      this.adjustWeights(
        epoch,
        this.layers[i],
        outputs[i],
        outputs[i + 1],
        errors[i],
        this.errorThreshold
      );
    }
  }

  private forward(inputs: Matrix): Matrix<"2D"> {
    const outputs: Matrix<"2D"> = [inputs];
    for (const layer of this.layers) {
      outputs.push(layer.forward(outputs[outputs.length - 1]));
    }
    return outputs;
  }

  private adjustWeights(
    epoch: number,
    layer: Layer,
    inputs: Matrix,
    outputs: Matrix,
    errors: Matrix,
    epsilon: number // Error threshold
  ): void {
    layer.neurons.forEach((neuron, i) => {
      // Check if the error is greater than epsilon
      if (Math.abs(errors[i]) > epsilon) {
        // Calculate the gradient only if the error is greater than epsilon
        const gradient: number =
          errors[i] *
          neuron.activationDerivative(outputs[i]) *
          this.learningRate;
        // Adjust the bias and weights only if the error exceeds epsilon
        neuron.bias += gradient;
        neuron.weights = neuron.weights.map(
          (weight, j) => weight + inputs[j] * gradient
        );
      } else {
        // this.log(
        //   `The error ${Math.abs(errors[i])} is less than the epsilon threshold (${epsilon}), no weights or bias adjustments for neuron ${neuron.identifier}.`
        // );
      }

      //Experimental feature: simulator
      if (this.saveHistory) {
        const data = {
          epoch,
          identifier: neuron.identifier,
          adjustWeights: {
            error: Math.abs(errors[i]),
            epsilon,
            value: Math.abs(errors[i]) > epsilon,
          },
          gradient: {
            error: errors[i],
            activationDerivative: neuron.activationDerivative(outputs[i]),
            learningRate: this.learningRate,
            value:
              errors[i] *
              neuron.activationDerivative(outputs[i]) *
              this.learningRate,
          },
          weights: neuron.weights.map((e) => e),
          bias: neuron.bias,
          inputs,
          outputs,
        };
        // if (Math.abs(errors[i]) > epsilon) {
        //   this.log(data);
        // }
        this.history.push(data);
      }
    });
  }

  private calculateErrors(outputErrors: Matrix): Matrix<"2D"> {
    const errors: Matrix<"2D"> = [outputErrors];
    for (let i = this.layers.length - 1; i > 0; i--) {
      const layer: Layer = this.layers[i];
      const previousLayer: Layer = this.layers[i - 1];
      const hiddenErrors: Matrix = previousLayer.neurons.map((neuron, j) =>
        errors[0].reduce(
          (sum, error, k) => sum + error * layer.neurons[k].weights[j],
          0
        )
      );
      errors.unshift(hiddenErrors);
    }
    return errors;
  }

  predict(inputs: Matrix): Matrix {
    return this.forward(inputs)[this.layers.length];
  }

  public saveTraining(): void {
    this.updateMetaData();
    const TrainingNetworkData: TrainingNetworkData = {
      metaData: this.metaData,
      weights: this.layers.map((layer) =>
        layer.neurons.map((neuron) => [...neuron.weights])
      ),
      biases: this.layers.map((layer) =>
        layer.neurons.map((neuron) => neuron.bias)
      ),
    };
    // Save the training to a JSON file
    NetworkStore.showLogs = this.showLogs;
    NetworkStore.saveObject(TrainingNetworkData, this.trainingName);
    this.log("Training saved successfully.");
  }

  public loadTraining(): boolean {
    // Load the training from a JSON file
    NetworkStore.showLogs = this.showLogs;
    const TrainingNetworkData: TrainingNetworkData | undefined =
      NetworkStore.loadObject(this.trainingName);
    if (!TrainingNetworkData) {
      this.log(`Training was not loaded.`);
      return false;
    }

    this.layers.forEach((layer, i) => {
      layer.neurons.forEach((neuron, j) => {
        neuron.weights = [...TrainingNetworkData.weights[i][j]];
        neuron.bias = TrainingNetworkData.biases[i][j];
      });
    });

    this.log(`Training loaded successfully.`);
    return true;
  }

  private log(...messages: any[]): void {
    if (this.showLogs) {
      console.log(...messages);
    }
  }
}
