'use client'

import React, { useState, useEffect, useRef } from 'react'

const MovieRecommender = () => {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { text: input, sender: 'user' }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('http://127.0.0.1:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input }),
      })
      const data = await response.json()
      const botMessage = { text: data.response, sender: 'bot' }
      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = { text: 'Something went wrong, please try again.', sender: 'bot' }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const styles = {
    container: {
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(to bottom right, #1a202c, #2a4365, #4c51bf)',
      padding: '1rem',
    },
    chatBox: {
      width: '100%',
      maxWidth: '56rem',
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      borderRadius: '0.5rem',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
      overflow: 'hidden',
      border: '1px solid #3182ce',
      backdropFilter: 'blur(8px)',
    },
    header: {
      padding: '1.5rem',
      background: 'linear-gradient(to right, #2563eb, #7c3aed)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    },
    title: {
      fontSize: '1.875rem',
      fontWeight: 'bold',
      color: 'white',
      display: 'flex',
      alignItems: 'center',
    },
    chatArea: {
      height: '60vh',
      overflowY: 'auto',
      padding: '1rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
    },
    message: {
      padding: '0.75rem',
      borderRadius: '0.375rem',
      maxWidth: '80%',
    },
    userMessage: {
      marginLeft: 'auto',
      backgroundColor: '#2563eb',
      color: 'white',
    },
    botMessage: {
      marginRight: 'auto',
      backgroundColor: '#7c3aed',
      color: 'white',
    },
    form: {
      padding: '1rem',
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      borderTop: '1px solid #3182ce',
    },
    input: {
      width: '100%',
      padding: '0.75rem',
      backgroundColor: '#1a202c',
      color: 'white',
      border: 'none',
      borderRadius: '0.375rem',
      marginRight: '0.5rem',
    },
    button: {
      padding: '0.75rem',
      background: 'linear-gradient(to right, #2563eb, #7c3aed)',
      color: 'white',
      fontWeight: 'bold',
      border: 'none',
      borderRadius: '0.375rem',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '1.25rem',
    },
    footer: {
      position: 'absolute',
      bottom: '1rem',
      right: '1rem',
      color: 'white',
      display: 'flex',
      alignItems: 'center',
    },
  }

  return (
    <div style={styles.container}>
      <div style={styles.chatBox}>
        <div style={styles.header}>
          <h1 style={styles.title}>
            <span style={{marginRight: '0.5rem'}}>✨</span>
            Movie Recommender ChatBot
          </h1>
          <span style={{color: 'white', fontSize: '1.875rem'}}>🎬</span>
        </div>
        <div style={styles.chatArea}>
          {messages.map((message, index) => (
            <div
              key={index}
              style={{
                ...styles.message,
                ...(message.sender === 'user' ? styles.userMessage : styles.botMessage),
              }}
            >
              {message.text}
            </div>
          ))}
          {isLoading && (
            <div style={{display: 'flex', justifyContent: 'center'}}>
              <span style={{color: '#60a5fa', fontSize: '1.5rem'}}>⏳</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={{display: 'flex'}}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              style={styles.input}
              placeholder="What's on your mind..."
            />
            <button
              type="submit"
              disabled={isLoading}
              style={{...styles.button, opacity: isLoading ? 0.5 : 1}}
            >
              {isLoading ? '⏳' : '↑'}
            </button>
          </div>
        </form>
      </div>
      <div style={styles.footer}>
        <span style={{marginRight: '0.5rem', fontSize: '1.5rem'}}>🍿</span>
        <span style={{fontSize: '0.875rem', fontWeight: 'semibold'}}>Powered by C-RECS</span>
      </div>
    </div>
  )
}

export default MovieRecommender
