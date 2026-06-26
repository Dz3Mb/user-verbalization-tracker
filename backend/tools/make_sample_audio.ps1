# Generate a local, entity-rich speech sample using the Windows TTS engine.
# Fully offline (no download) — useful to test transcription + entity linking
# in a reproducible way.
#
# Usage (from the backend folder):
#   powershell -ExecutionPolicy Bypass -File tools/make_sample_audio.ps1
#
# Output: uploads/sample_speech.wav

Add-Type -AssemblyName System.Speech

$outDir = Join-Path $PSScriptRoot "..\uploads"
$outFile = Join-Path $outDir "sample_speech.wav"

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = -1  # slightly slower for cleaner transcription

$text = @"
Barack Obama was born in Hawaii and served as President of the United States.
Microsoft and Apple are technology companies based in California.
The Eiffel Tower is located in Paris, the capital of France.
"@

$synth.SetOutputToWaveFile($outFile)
$synth.Speak($text)
$synth.Dispose()

Write-Output "Sample written to: $outFile"
