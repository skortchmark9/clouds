// jest.config.js
module.exports = {
    // ...
    preset: 'jest-puppeteer',
    moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
    transform: {
      '^.+\\.tsx?$': 'ts-jest', // Use ts-jest to handle TypeScript files
    },  
  }
