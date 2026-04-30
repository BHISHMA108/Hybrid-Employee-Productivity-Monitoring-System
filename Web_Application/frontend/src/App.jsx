import { useState } from 'react';
import ThemeToggle from './components/ThemeToggle';
import ModelCard from './components/ModelCard';
import AccuracyComparison from './components/AccuracyComparison';
import ImageUploader from './components/ImageUploader';

export default function App() {
  const [predictions, setPredictions] = useState(null);

  const models = [
    {
      id: "pytorch",
      title: "PyTorch Model",
      framework: "MobileNetV2 / PyTorch",
      accuracy: 89.40,
      precision: 0.88,
      recall: 0.87,
      f1Score: 0.875,
      latency: "45ms",
      description: "A lightweight MobileNetV2 architecture trained from scratch using PyTorch. Optimized for mobile and edge deployment, this model serves as our baseline for performance comparison in resource-constrained environments."
    },
    {
      id: "keras",
      title: "Keras Model",
      framework: "InceptionV3 / TensorFlow",
      accuracy: 98.30,
      precision: 0.97,
      recall: 0.98,
      f1Score: 0.975,
      latency: "120ms",
      description: "A high-capacity TensorFlow implementation using InceptionV3. This model leverages deep feature extraction to achieve high precision on complex employee activity datasets, though with higher computational overhead."
    },
    {
      id: "student",
      title: "Student Model",
      framework: "Hybrid Distillation (Teacher-Student)",
      accuracy: 99.00,
      precision: 0.99,
      recall: 0.99,
      f1Score: 0.99,
      latency: "32ms",
      description: "The final optimized hybrid model. By distilling knowledge from larger ensembles into a streamlined architecture, it achieves state-of-the-art 99.00% accuracy with ultra-low latency for real-time tracking."
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300 font-sans selection:bg-indigo-500 selection:text-white">
      {/* Abstract Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-indigo-400/20 dark:bg-indigo-900/20 blur-3xl"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-purple-400/20 dark:bg-purple-900/20 blur-3xl"></div>
      </div>

      <div className="relative z-10">
        {/* Header with Fixed Z-Index */}
        <header className="px-6 py-6 md:px-12 md:py-8 flex justify-between items-start md:items-center border-b border-gray-200/50 dark:border-gray-800/50 bg-white/30 dark:bg-black/30 backdrop-blur-md sticky top-0 z-50">
          <div className="relative z-50">
            <h1 className="text-2xl md:text-3xl font-black bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-400 dark:to-purple-400 max-w-4xl leading-tight">
              Employee Productivity Monitoring System
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2 font-medium text-sm md:text-base">
              Using a Hybrid Deep Learning-Based Detection and Activity Tracking
            </p>
          </div>
          <div className="ml-4 mt-1 md:mt-0 flex-shrink-0 relative z-50">
            <ThemeToggle />
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-6 py-12 md:py-20">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
              Model Evaluation & Performance
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
              Our system evaluates employee engagement through advanced computer vision. 
              By comparing PyTorch and Keras backbones, we developed a <strong>Student-Teacher Hybrid </strong> 
              that maximizes inference speed without compromising on classification accuracy.
            </p>
          </div>

          <div className="flex flex-col items-center justify-center mb-16">
            <ImageUploader onPredictionsUpdate={setPredictions} />
          </div>

          {/* Model Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-20">
            {models.map((model) => (
              <ModelCard 
                key={model.id}
                title={model.title}
                framework={model.framework}
                accuracy={model.accuracy}
                description={model.description}
                prediction={predictions ? predictions[model.id] : null}
                // Passing extra metrics if your ModelCard component supports them
                metrics={{
                    precision: model.precision,
                    recall: model.recall,
                    latency: model.latency
                }}
              />
            ))}
          </div>

          {/* Comparison Section */}
          <div className="max-w-4xl mx-auto">
            <AccuracyComparison models={models} />
          </div>
        </main>
        
        <footer className="py-12 px-6 text-gray-600 dark:text-gray-400 border-t border-gray-200/50 dark:border-gray-800/50 mt-12 bg-white/10 dark:bg-black/10 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto flex flex-col items-center gap-6">
            <div className="flex flex-wrap justify-center gap-x-8 gap-y-2 font-semibold text-gray-800 dark:text-gray-200">
              <span>Uday Dandekar</span>
              <span>Megha Derkar</span>
              <span>Shivam Dhokane</span>
            </div>
            
            <div className="text-indigo-600 dark:text-indigo-400 font-bold tracking-widest uppercase text-xs md:text-sm">
              Final Year Sem 2 Major Project
            </div>

            <div className="text-xs text-gray-500">
              &copy; {new Date().getFullYear()} EfficienSee Project. All rights reserved.
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}