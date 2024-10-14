/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import * as fs from "fs";
import * as path from "path";

export class StorageManager<T> {
  private filePath: string;

  private _showLogs: boolean = false;
  public get showLogs(): boolean {
    return this._showLogs;
  }
  public set showLogs(value: boolean) {
    this._showLogs = value;
  }

  constructor(filePath: string = "./training") {
    this.filePath = filePath;
  }

  private isNode(): boolean {
    return (
      typeof process !== "undefined" &&
      process.versions != null &&
      process.versions.node != null
    );
  }

  public saveObject(
    obj: T,
    fileName: string,
    formatted: boolean = false
  ): void {
    // Convert the object to a JSON string with or without pretty-printing based on the formatted parameter
    const jsonData = formatted
      ? JSON.stringify(obj, null, 2)
      : JSON.stringify(obj);

    if (this.isNode()) {
      const fullPath = path.join(this.filePath, `${fileName}.json`);
      const dirPath = path.dirname(fullPath);

      // Check if directory exists, if not create it
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
      }

      fs.writeFileSync(fullPath, jsonData, "utf8");
      this.log(`Object saved to ${fullPath}`);
    } else {
      localStorage.setItem(`${this.filePath}/${fileName}`, jsonData);
      this.log(
        `Object saved to localStorage with key ${this.filePath}/${fileName}`
      );
    }
  }

  public loadObject(fileName: string): T {
    if (this.isNode()) {
      const fullPath = path.join(this.filePath, `${fileName}.json`);
      const jsonData = fs.readFileSync(fullPath, "utf8");
      const obj: T = JSON.parse(jsonData);
      this.log(`Object loaded from ${fullPath}`);
      return obj;
    } else {
      const jsonData = localStorage.getItem(`${this.filePath}/${fileName}`);
      if (!jsonData) {
        throw new Error(
          `No data found in localStorage for key ${this.filePath}/${fileName}`
        );
      }
      const obj: T = JSON.parse(jsonData);
      this.log(
        `Object loaded from localStorage with key ${this.filePath}/${fileName}`
      );
      return obj;
    }
  }

  private log(...messages: any[]): void {
    if (this.showLogs) {
      console.log(...messages);
    }
  }
}
