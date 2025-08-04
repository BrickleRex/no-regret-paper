import sys
import os
import datetime
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, log_dir="logs", tag="", writer=False):
        """
        Initialize the Logger by creating a timestamped folder.
        Redirects sys.stdout and sys.stderr so that all print statements
        are written both to the console and to a log file.
        """
        # Create a timestamp for this run.
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_folder = os.path.join(log_dir, f"run_{timestamp}_{tag}")

        if writer:
            os.makedirs(self.run_folder, exist_ok=True)
        
        # Create the log file within the run folder.
        self.log_file_path = os.path.join(self.run_folder, "run.log")

        if writer:
            self.log_file = open(self.log_file_path, "w", buffering=1)
        
        # # Save original stdout and stderr.
        # self.original_stdout = sys.stdout
        # self.original_stderr = sys.stderr

        self.writer = writer
        
        # Redirect stdout and stderr to this logger instance.
        # sys.stdout = self
        # sys.stderr = self

    def write(self, message):
        """
        Write the message both to the original stdout (console) and the log file.
        """
        # Only add newline if message doesn't already end with one
        if not message.endswith('\n'):
            message = message + '\n'
            
        print(message)
        
        # self.original_stdout.write(message)

        if self.writer:
            self.log_file.write(message)

    def flush(self):
        """
        Flush both the console and the file streams.
        """
        self.original_stdout.flush()

        if self.writer:
            self.log_file.flush()

    def log_variable(self, var_name, value):
        """
        Logs a variable's name and value with a timestamp.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {var_name}: {value}\n"
        self.write(log_message)

    def save_plot(self, fig, plot_name):
        """
        Saves the provided matplotlib figure into the run folder with the given plot name.
        """
        if not self.writer:
            return
        
        file_path = os.path.join(self.run_folder, f"{plot_name}.png")
        fig.savefig(file_path)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write(f"[{timestamp}] Plot saved: {file_path}\n")
        
    def write_allocations(self, allocations_df, tag=""):
        """
        Writes the allocations to a CSV file.
        """
        file_path = os.path.join(self.run_folder, f"{tag}allocations.csv")
        allocations_df.to_csv(file_path, index=False)            
        print(f"Allocations written to {file_path}")
        
    def write_results(self, results_df, tag=""):
        """
        Writes the results to a CSV file.
        """
        file_path = os.path.join(self.run_folder, f"{tag}totalval_results.csv")
        results_df.to_csv(file_path)
        print(f"Results written to {file_path}")

    def get_log_dir(self):
        """
        Returns the directory where the logs are stored.
        """
        return self.run_folder

    def close(self):
        """
        Resets sys.stdout and sys.stderr to their original streams and closes the log file.
        """
        # sys.stdout = self.original_stdout
        # sys.stderr = self.original_stderr

        if self.writer:
            self.log_file.close()

# # Example usage:
# if __name__ == "__main__":
#     # Create an instance of the Logger.
#     logger = Logger()

#     # Now, every print statement will go both to the console and the log file.
#     print("Starting the run...")

#     # Log a variable.
#     x = 42
#     logger.log_variable("x", x)

#     # Create a simple plot and save it.
#     fig, ax = plt.subplots()
#     ax.plot([1, 2, 3], [1, 4, 9])
#     logger.save_plot(fig, "sample_plot")

#     print("Run complete.")

#     # Reset stdout/stderr and close the log file when done.
#     logger.close()
