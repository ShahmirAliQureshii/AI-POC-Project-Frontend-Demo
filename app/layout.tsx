// @ts-ignore: allow side-effect CSS import without types
import "./globals.css";
import { Inter } from "next/font/google";
import type { Metadata } from "next";
import React from "react";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Face Recognition • Modern UI",
  description: "Modern frontend for Face Recognition System (no backend)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} font-sans bg-gradient-to-br from-slate-50 via-slate-100 to-slate-50 text-slate-900 min-h-screen`}
      >
        {children}
      </body>
    </html>
  );
}