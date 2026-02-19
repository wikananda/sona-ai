import AudioUploader from "@/src/components/AudioUploader"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className="text-3xl font-bold">Sona AI</h1>
      <AudioUploader />
    </main>
  );
}