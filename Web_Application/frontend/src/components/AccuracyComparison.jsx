export default function AccuracyComparison({ models }) {
  // Sort models by accuracy
  const sortedModels = [...models].sort((a, b) => b.accuracy - a.accuracy);

  return (
    <div className="bg-white/50 dark:bg-gray-800/50 backdrop-blur-md rounded-3xl p-8 shadow-2xl border border-white/20 dark:border-gray-700/30">
      <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-8 text-center">
        Performance Comparison
      </h2>
      
      <div className="space-y-6">
        {sortedModels.map((model, idx) => (
          <div key={idx} className="relative">
            <div className="flex justify-between items-center mb-2">
              <span className="font-semibold text-gray-700 dark:text-gray-200">
                {model.title}
              </span>
              <span className="font-bold text-indigo-600 dark:text-indigo-400">
                {model.accuracy}%
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden shadow-inner">
              <div 
                className={`h-4 rounded-full ${
                  idx === 0 ? 'bg-gradient-to-r from-emerald-400 to-emerald-600' :
                  idx === 1 ? 'bg-gradient-to-r from-blue-400 to-indigo-600' :
                  'bg-gradient-to-r from-purple-400 to-pink-600'
                }`}
                style={{ width: `${model.accuracy}%`, transition: 'width 1.5s ease-in-out' }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
