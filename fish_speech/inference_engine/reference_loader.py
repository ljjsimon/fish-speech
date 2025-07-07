import io
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal, Tuple

import torch
import torchaudio
from loguru import logger
import numpy as np

from fish_speech.models.dac.modded_dac import DAC
from fish_speech.utils.file import (
    AUDIO_EXTENSIONS,
    audio_to_bytes,
    list_files,
    read_ref_text,
)
from fish_speech.utils.schema import ServeReferenceAudio


class ReferenceLoader:

    def __init__(self) -> None:
        """
        Component of the TTSInferenceEngine class.
        Loads and manages the cache for the reference audio and text.
        """
        self.ref_by_id: dict = {}
        self.ref_by_hash: dict = {}

        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.encode_reference: Callable

        # Define the torchaudio backend
        backends = torchaudio.list_audio_backends()
        if "ffmpeg" in backends:
            self.backend = "ffmpeg"
        else:
            self.backend = "soundfile"

    def load_by_id(
        self,
        id: str,
        use_cache: Literal["on", "off"],
    ) -> Tuple:

        # Load the references audio and text by id
        ref_folder = Path("references") / id
        ref_folder.mkdir(parents=True, exist_ok=True)
        ref_audios = list_files(
            ref_folder, AUDIO_EXTENSIONS, recursive=True, sort=False
        )

        if use_cache == "off" or id not in self.ref_by_id:
            # If the references are not already loaded, encode them
            prompt_tokens = [
                self.encode_reference(
                    # decoder_model=self.decoder_model,
                    reference_audio=audio_to_bytes(str(ref_audio)),
                    enable_reference_audio=True,
                )
                for ref_audio in ref_audios
            ]
            prompt_texts = [
                read_ref_text(str(ref_audio.with_suffix(".lab")))
                for ref_audio in ref_audios
            ]
            self.ref_by_id[id] = (prompt_tokens, prompt_texts)

        else:
            # Reuse already encoded references
            logger.info("Use same references")
            prompt_tokens, prompt_texts = self.ref_by_id[id]

        return prompt_tokens, prompt_texts

    def load_by_hash(
        self,
        references: list[ServeReferenceAudio],
        use_cache: Literal["on", "off"],
        save_prompt: bool = True,
    ) -> Tuple:

        # Load the references audio and text by hash
        audio_hashes = [sha256(ref.audio).hexdigest() for ref in references]

        cache_used = False
        prompt_tokens, prompt_texts = [], []
        for i, ref in enumerate(references):
            if use_cache == "off" or audio_hashes[i] not in self.ref_by_hash:
                # If the references are not already loaded, encode them
                prompt_tokens.append(
                    self.encode_reference(
                        reference_audio=ref.audio,
                        enable_reference_audio=True,
                    )
                )
                prompt_texts.append(ref.text)
                self.ref_by_hash[audio_hashes[i]] = (prompt_tokens[-1], ref.text)

                if save_prompt:
                    self.save_prompt(
                        ref = ref,
                        prompt_token_npy = prompt_tokens[-1].cpu().numpy(),
                        ref_id=audio_hashes[i],
                    )
                    logger.info(f"Saved prompt with id: {audio_hashes[i]}")
            else:
                # Reuse already encoded references
                cached_token, cached_text = self.ref_by_hash[audio_hashes[i]]
                prompt_tokens.append(cached_token)
                prompt_texts.append(cached_text)
                cache_used = True

        if cache_used:
            logger.info("Use same references")

        return prompt_tokens, prompt_texts

    def load_audio(self, reference_audio, sr):
        """
        Load the audio data from a file or bytes.
        """
        if len(reference_audio) > 255 or not Path(reference_audio).exists():
            audio_data = reference_audio
            reference_audio = io.BytesIO(audio_data)

        waveform, original_sr = torchaudio.load(reference_audio, backend=self.backend)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if original_sr != sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=sr
            )
            waveform = resampler(waveform)

        audio = waveform.squeeze().numpy()
        return audio
    
    def save_prompt(
        self,
        ref: ServeReferenceAudio,
        prompt_token_npy: np.ndarray,
        ref_id: str,
    ) -> None:
        """
        Save the prompt tokens and texts to a file.
        """
        ref_folder = Path("references") / ref_id
        ref_folder.mkdir(parents=True, exist_ok=True)

        # Save the reference audios and texts
        ref_audio_path = ref_folder / f"audio.wav"
        # save prompt tokens
        np.save(ref_audio_path.with_suffix(".npy"), prompt_token_npy)
        # save aduio
        with open(ref_audio_path, "wb") as f:
            f.write(ref.audio)
        # save prompt text
        with open(ref_audio_path.with_suffix(".lab"), "w", encoding="utf-8") as f:
            f.write(ref.text + "\n")

    def fast_load_by_id(
        self,
        id: str,
        use_cache: Literal["on", "off"],
    ) -> Tuple[list, list]:
        # Load the references audio and text by id
        ref_folder = Path("references") / id
        ref_folder.mkdir(parents=True, exist_ok=True)
        ref_audios = list_files(
            ref_folder, {"npy"}, recursive=True, sort=False
        )

        if use_cache == "off" or id not in self.ref_by_id:
            # If the references are not already loaded, encode them
            prompt_tokens = [
                torch.from_numpy(np.load(str(ref_audio)))
                for ref_audio in ref_audios
            ]
            prompt_texts = [
                read_ref_text(str(ref_audio.with_suffix(".lab")))
                for ref_audio in ref_audios
            ]
            self.ref_by_id[id] = (prompt_tokens, prompt_texts)

        else:
            # Reuse already encoded references
            logger.info("Use same references")
            prompt_tokens, prompt_texts = self.ref_by_id[id]

        return prompt_tokens, prompt_texts
    
