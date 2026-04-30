export default function ModelCard({ title, framework, accuracy, description, prediction }) {
  return (
    <div className="bg-white/70 dark:bg-gray-800/70 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-gray-200/50 dark:border-gray-700/50 hover:scale-[1.02] transition-transform duration-300 flex flex-col items-center text-center">
      <div className="mb-4">
        <h3 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white">
          {title}
        </h3>
        <p className="text-sm font-medium text-indigo-600 dark:text-indigo-400 mt-1 uppercase tracking-wider">
          {framework}
        </p>
      </div>
      
      <p className="text-gray-600 dark:text-gray-300 text-sm mb-6 flex-grow">
        {description}
      </p>

      {/* Prediction Result Section */}
      {prediction && (
        <div className="w-full bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-4 mb-6 border border-indigo-100 dark:border-indigo-800/50">
          <p className="text-xs text-indigo-500 dark:text-indigo-400 font-bold uppercase tracking-widest mb-1">Live Prediction</p>
          <div className="flex justify-between items-center">
            <span className={`text-lg font-black ${prediction.class === 'Face' ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
              {prediction.class}
            </span>
            <span className="text-sm font-bold text-gray-700 dark:text-gray-300">
              {(prediction.confidence * 100).toFixed(2)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5 mt-2 overflow-hidden">
            <div 
              className={`h-1.5 rounded-full ${prediction.class === 'Face' ? 'bg-emerald-500' : 'bg-rose-500'}`}
              style={{ width: `${prediction.confidence * 100}%` }}
            ></div>
          </div>
        </div>
      )}

      <div className="mt-auto w-full">
        <div className="flex justify-between items-end mb-2">
          <span className="text-sm font-semibold text-gray-500 dark:text-gray-400">Baseline Accuracy</span>
          <span className="text-2xl font-black text-gray-900 dark:text-white">{accuracy}%</span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 overflow-hidden">
          <div 
            className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2.5 rounded-full" 
            style={{ width: `${accuracy}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
}
