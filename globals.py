import matplotlib.pyplot as plt
import scipy.signal as signal

### test data, For good inference you need 5 hours of music for each genre
### from each song the trainin get's train_chunks_per_file  chunks of duration_sec_train
### startgin from the begingin of the file     

#                                   / -number of files to download for this genere
#                    / - yt query  |  /- enable|disable [1|0] this download and training for this genre 
#                   |              | |
GENERES = {'all': 'xyzabcd-fgh-ijk|0|1',  # keep this always 0. for tetgin all the files
           'insAI':'instrumental metal and heavy metal AI music|100|0',
           'amon': 'amon amarth|40|0',
           'goodbye': 'goodbye to gravity|30|0',
           'infected': 'infected rain instrumetal|20|0',
           'accept': 'accept instrumental|20|0',
           'phoenix': 'phoenix romania|20|0'}



def gen_path(p):
    return f'./DNL/{p}/'

def gen_proc(p):
    return f'./DNL/{p}/proc'

def gen_bps(p):
    return f'./DNL/{p}/rate'

def vae_path(p):
    """Final VQ-VAE weights for a genre."""
    return f'M/{p}-vqvae.pt'

def vae_chk_path(p):
    """Resumable training checkpoint for a genre."""
    return f'M/{p}-vqvae-checkpoint.pt'

def vae_best_path(p):
    """Best-loss VQ-VAE weights for a genre."""
    return f'M/{p}-vqvae-best.pt'

def gen_model_path(p):
    """Final VQ-VAE weights for a genre."""
    return f'M/{p}-gen.pt'

def gen_chk_path(p):
    """Final VQ-VAE weights for a genre."""
    return f'M/{p}-gen-checkpoint.pt'
    

class Ploter:
    def __init__(self):
        self.epoch_total_losses = []
        self.epoch_kl_losses = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        
        # Plot lines for both Total Loss and VQ / KL Commitment Loss
        self.line_total, = self.ax.plot([], [], marker='o', linestyle='-', color='b', label='Total Loss')
        self.line_kl, = self.ax.plot([], [], marker='s', linestyle='--', color='r', label='VQ / KL Loss')
        
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Live VAE Loss Plot')
        self.ax.grid(True)
        self.ax.legend()

    def add_point(self, epoch_total_loss, epoch_kl_loss=0.0):
        # Skip non-finite values so the plot never crashes
        if epoch_total_loss is None or not (epoch_total_loss == epoch_total_loss) or abs(epoch_total_loss) == float("inf"):
            print(f"[Ploter] Skipping non-finite loss: {epoch_total_loss}")
            return

        kl_val = float(epoch_kl_loss) if (epoch_kl_loss is not None and epoch_kl_loss == epoch_kl_loss and abs(epoch_kl_loss) != float("inf")) else 0.0

        self.epoch_total_losses.append(float(epoch_total_loss))
        self.epoch_kl_losses.append(kl_val)

        if not self.epoch_total_losses:
            return

        epochs_range = list(range(1, len(self.epoch_total_losses) + 1))
        
        # Update line data for both metrics
        self.line_total.set_data(epochs_range, self.epoch_total_losses)
        self.line_kl.set_data(epochs_range, self.epoch_kl_losses)
        
        self.ax.set_xlim(1, max(epochs_range) + 1)
        
        # Rescale y-axis dynamically based on both loss curves
        all_vals = [v for v in (self.epoch_total_losses + self.epoch_kl_losses) if v == v and abs(v) != float("inf")]
        if all_vals:
            ymin = min(all_vals) * 0.9
            ymax = max(all_vals) * 1.1
            if ymin == ymax:
                ymax = ymin + 1e-6
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            self.ax.set_ylim(ymin, ymax)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def __del__(self):
        try:
            plt.ioff()
            plt.close(self.fig)
        except Exception:
            pass


        