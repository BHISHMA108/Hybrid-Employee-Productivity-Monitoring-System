import { useState, useRef } from 'react';

export default function ImageUploader({ onPredictionsUpdate }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to get predictions from the server.');
      }

      const data = await response.json();
      onPredictionsUpdate(data);
    } catch (err) {
      setError(err.message);
      onPredictionsUpdate(null);
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setImage(null);
    setPreview(null);
    setError(null);
    onPredictionsUpdate(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="bg-white/70 dark:bg-gray-800/70 backdrop-blur-md rounded-2xl p-8 shadow-xl border border-gray-200/50 dark:border-gray-700/50 mb-12 flex flex-col items-center">
      <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Test the Models</h3>
      
      {!preview ? (
        <div 
          className="w-full max-w-md border-2 border-dashed border-indigo-300 dark:border-indigo-700 rounded-xl p-12 text-center cursor-pointer hover:bg-indigo-50 dark:hover:bg-indigo-900/20 transition-colors"
          onClick={() => fileInputRef.current.click()}
        >
          <svg className="mx-auto h-12 w-12 text-indigo-400 dark:text-indigo-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <p className="text-gray-600 dark:text-gray-400 font-medium">Click to upload an image</p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">JPEG, PNG, JPG</p>
        </div>
      ) : (
        <div className="w-full max-w-md flex flex-col items-center">
          <div className="relative w-full h-64 mb-6 rounded-xl overflow-hidden shadow-lg border border-gray-200 dark:border-gray-700">
            <img src={preview} alt="Preview" className="w-full h-full object-cover" />
            <button 
              onClick={clearImage}
              className="absolute top-3 right-3 bg-red-500/80 hover:bg-red-600 text-white rounded-full p-2 backdrop-blur-sm transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
          
          <button
            onClick={handleUpload}
            disabled={loading}
            className={`w-full py-3 px-6 rounded-xl font-bold text-white shadow-lg transition-all transform hover:scale-[1.02] active:scale-95 ${
              loading 
                ? 'bg-indigo-400 dark:bg-indigo-600 cursor-not-allowed' 
                : 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700'
            }`}
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing Models...
              </span>
            ) : (
              'Run Inference'
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-lg text-sm w-full max-w-md border border-red-200 dark:border-red-800">
          <p className="font-bold mb-1">Error running inference:</p>
          <p>{error}</p>
        </div>
      )}

      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleImageChange} 
        accept="image/*" 
        className="hidden"
      />
    </div>
  );
}
