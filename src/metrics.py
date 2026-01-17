def metrics(
    seq_len: int,
    autocorrelation = True,
):
    
    def run(
        ages, 
        features,
    ):
    
        seq_features = np.array(features) / np.mean(np.abs(features))
        
        variance = np.array([np.var(seq_features[i-seq_len:i]) for i in range(seq_len, len(seq_features))])
        
        autocorrelation = None if not autocorrelation else None
        
        def print_results(data_name: str, age_format: str):
            pass
        
        def plot_results(data_name: str, age_format: str):
                
            # Variance plot
            
            varfig, axs = plt.subplot(1, 1, figsize=(8,5), sharex = True)
            varfig.suptitle("Variance")
            varfig.plot(ages[seq_len:], variance)
            
            # Auto-correlation plot
            
            if autocorrelation is not None:
                acfig, axs = plt.subplot(2, 1, figsize=(8,5), sharex = True)
                acfig.suptitle("Autocorrelation")
                   
        
        return (print_results, plot_results)
    
    return ("Metric-based analysis", run)