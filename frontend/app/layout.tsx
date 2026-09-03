import type { Metadata } from "next";
import ProcessingActivityProvider from "@/src/components/ProcessingActivityProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sona AI",
  description: "Project-based transcription workspace",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <ProcessingActivityProvider>{children}</ProcessingActivityProvider>
      </body>
    </html>
  );
}
