import React, { useEffect, useState } from "react";
import {
  addEmergencyContact,
  getEmergencyContacts,
  deleteEmergencyContact,
} from "../api/emergencyApi";
import "./emergency.css";

export default function Emergency() {
  const [contacts, setContacts] = useState([]);
  const [loading, setLoading] = useState(true);

  const [form, setForm] = useState({
    name: "",
    phone: "",
    relation: "",
  });

  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  // Load contacts
  const loadContacts = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await getEmergencyContacts();

      setContacts(response.contacts || []);
    } catch (err) {
      setError(err.message || "Unable to load contacts");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadContacts();
  }, []);

  // Input change
  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  // Add contact
  const handleSubmit = async (e) => {
    e.preventDefault();

    setMessage("");
    setError("");

    if (!form.name || !form.phone) {
      setError("Name and phone number are required.");
      return;
    }

    try {
      await addEmergencyContact({
        name: form.name,
        phone: form.phone,
        relation: form.relation || null,
      });

      setMessage("Emergency contact added successfully.");

      setForm({
        name: "",
        phone: "",
        relation: "",
      });

      await loadContacts();
    } catch (err) {
      setError(
        err.message || "Failed to add emergency contact"
      );
    }
  };

  // Delete contact
  const handleDelete = async (id) => {
    const confirmDelete = window.confirm(
      "Are you sure you want to delete this contact?"
    );

    if (!confirmDelete) return;

    try {
      setError("");
      setMessage("");

      await deleteEmergencyContact(id);

      setMessage("Emergency contact deleted successfully.");

      await loadContacts();
    } catch (err) {
      setError(
        err.message || "Failed to delete contact"
      );
    }
  };

  return (
    <div className="emergency-page">

      {/* Header */}
      <div className="emergency-header">
        <div>
          <p className="page-label">GUARDIA SAFETY</p>
          <h1>Emergency Contacts</h1>
          <p className="page-subtitle">
            People who can be contacted when you need help.
          </p>
        </div>

        <div className="safety-badge">
          <span>●</span>
          Safety Ready
        </div>
      </div>

      {/* Messages */}
      {message && (
        <div className="success-message">
          ✓ {message}
        </div>
      )}

      {error && (
        <div className="error-message">
          ⚠ {error}
        </div>
      )}

      <div className="emergency-content">

        {/* Add Contact Card */}
        <div className="contact-card add-card">
          <div className="card-title">
            <div className="title-icon">+</div>

            <div>
              <h2>Add Emergency Contact</h2>
              <p>
                Add someone you trust for emergency situations.
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit}>

            <div className="form-group">
              <label>Full Name</label>

              <input
                type="text"
                name="name"
                value={form.name}
                onChange={handleChange}
                placeholder="Enter contact name"
              />
            </div>

            <div className="form-group">
              <label>Phone Number</label>

              <input
                type="tel"
                name="phone"
                value={form.phone}
                onChange={handleChange}
                placeholder="Enter phone number"
              />
            </div>

            <div className="form-group">
              <label>Relation</label>

              <input
                type="text"
                name="relation"
                value={form.relation}
                onChange={handleChange}
                placeholder="e.g. Father, Mother, Friend"
              />
            </div>

            <button
              type="submit"
              className="add-contact-btn"
            >
              + Add Contact
            </button>

          </form>
        </div>

        {/* Contact List */}
        <div className="contact-card list-card">

          <div className="list-header">
            <div>
              <h2>Your Emergency Contacts</h2>

              <p>
                {contacts.length} contact
                {contacts.length !== 1 ? "s" : ""} saved
              </p>
            </div>

            <div className="contact-count">
              {contacts.length}
            </div>
          </div>

          {loading ? (
            <div className="empty-state">
              <div className="loader"></div>
              <p>Loading contacts...</p>
            </div>
          ) : contacts.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">👥</div>

              <h3>No emergency contacts</h3>

              <p>
                Add at least one trusted person
                for emergency situations.
              </p>
            </div>
          ) : (
            <div className="contacts-list">

              {contacts.map((contact) => (
                <div
                  className="contact-item"
                  key={contact._id || contact.id}
                >

                  <div className="contact-avatar">
                    {contact.name
                      ?.charAt(0)
                      ?.toUpperCase()}
                  </div>

                  <div className="contact-info">
                    <h3>{contact.name}</h3>

                    <p>
                      📞 {contact.phone}
                    </p>

                    {contact.relation && (
                      <span>
                        {contact.relation}
                      </span>
                    )}
                  </div>

                  <button
                    className="delete-btn"
                    onClick={() =>
                      handleDelete(
                        contact._id || contact.id
                      )
                    }
                    title="Delete contact"
                  >
                    🗑
                  </button>

                </div>
              ))}

            </div>
          )}

        </div>

      </div>

      {/* Safety information */}
      <div className="safety-info">
        <div className="info-icon">🛡</div>

        <div>
          <h3>Why add emergency contacts?</h3>

          <p>
            When an SOS is triggered, these contacts can
            be used to help coordinate assistance and
            share your emergency location.
          </p>
        </div>
      </div>

    </div>
  );
}