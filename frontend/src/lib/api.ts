import axios from 'axios';
import { Song } from '../types/song';

const apiBase = import.meta.env.VITE_API_URL ?? '/api';

const http = axios.create({ baseURL: apiBase });

export async function uploadSong(
  file: File
): Promise<{ id: string; status: string }> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await http.post('/songs', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function getSong(id: string): Promise<Song> {
  const { data } = await http.get<Song>(`/songs/${id}`);
  return data;
}

export async function listSongs(): Promise<Song[]> {
  const { data } = await http.get<Song[]>('/songs');
  return data;
}

export async function deleteSong(id: string): Promise<void> {
  await http.delete(`/songs/${id}`);
}

export function getMixAudioUrl(songId: string): string {
  return `${apiBase}/songs/${songId}/audio`;
}

export function getStemAudioUrl(songId: string, stem: string): string {
  return `${apiBase}/songs/${songId}/stems/${stem}`;
}
